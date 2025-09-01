from sympy import Q
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 这里program_id(0)和program_id(1)分别对应启动triton kernel的grid的顺序
    i = tl.program_id(0) #i就代表Q的第i个tile
    batch_index = tl.program_id(1)

    # 这里offset每个指针，对应batch index乘以batch stride
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), #注意K，V的初始化偏移是0，它们会在后续的循环中被更新
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    #对于因果掩码，主要针对Q和K的矩阵乘法，最终目标是求得S矩阵的上三角区域的值变成-inf
    #在FLashAttention操作中，目标仍然是求大的S矩阵的上三角区域的值变成-inf，但因为是分块求的S_ij，所以如果i<j即代表S_ij处在大S矩阵的上三角区域，所以需要把S_ij变成-inf;如果i>j即代表S_ij处在大S矩阵的下三角区域，不用管。
    #一类特殊情况就是i=j，此时的S_ij正好处在对角上，这里的小S_ij也要去把上三角给变成-inf

    Q_i = tl.load(Q_block_ptr) #(Q_TILE_SIZE, D)
    O_i_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) #(Q_TILE_SIZE, D)
    L_i_acc = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32) #(Q_TILE_SIZE, 1)
    M_i_acc = tl.full((Q_TILE_SIZE, 1), float('-inf'), dtype=tl.float32) #(Q_TILE_SIZE, 1)
    # 外层循环：对 K/V 的序列长度进行分块
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr) #(K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr) #(K_TILE_SIZE, D)
        # 计算attention scores
        S_ij = tl.dot(Q_i, K_j.T) * scale #(Q_TILE_SIZE, K_TILE_SIZE)
        
        # 应用因果掩码 - 使用一个统一的掩码方法
        if is_causal:
        #     if i < j: #i<j即代表S_ij处在大S矩阵的上三角区域，所以需要把S_ij变成-inf
        #         S_ij = tl.full((Q_TILE_SIZE, K_TILE_SIZE), -1e6, dtype=tl.float32)
        #     elif j == i:#如果j==i，则代表S_ij正好处在对角上，这里的小S_ij也要去把上三角给变成-inf
        #         mask = tl.arange(0,Q_TILE_SIZE)[:, None] >= tl.arange(0, K_TILE_SIZE)[None, :]
        #         S_ij = tl.where(mask, S_ij, -1e6)
            #创建位置掩码,在triton里面必须要用这种写法，因为triton在编译的时候是不通过if else的。
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None] #(Q_TILE_SIZE, 1) #第一项实际上就是偏移的位置了
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :] #(1, K_TILE_SIZE)
            causal_mask = q_idx >= k_idx #(Q_TILE_SIZE, K_TILE_SIZE) 这里比较的思路就是把每次的偏移之后的行、列索引都进行比较，如果q_idx>=k_idx，则代表S_ij处在大S矩阵的上三角区域，所以需要把S_ij变成-inf
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        M_ij = tl.max(S_ij, axis=1, keep_dims=True) #(Q_TILE_SIZE, 1) - 当前块的最大值
        M_i_new = tl.maximum(M_i_acc, M_ij) #(Q_TILE_SIZE, 1) - 与累积最大值比较
        P_ij = tl.exp(S_ij - M_ij) #(Q_TILE_SIZE, K_TILE_SIZE) - 注意：使用M_ij而不是M_i_new
        L_i_new = tl.exp(M_i_acc - M_i_new) * L_i_acc + tl.exp(M_ij - M_i_new) * tl.sum(P_ij, axis=1, keep_dims=True) #(Q_TILE_SIZE, 1)
        #数据对齐
        P_ij_cast = P_ij.to(V_j.dtype)#这代表从指针全局内存中读取类型，用V_j.dype理论上也行。
        O_i_new = tl.exp(M_i_acc - M_i_new) * O_i_acc + tl.exp(M_ij - M_i_new) * tl.dot(P_ij_cast, V_j)  #(Q_TILE_SIZE, D)

        #与pytorch代码不同，这里必须显式地更新M_i_acc和O_i_acc，因为下一个循环不会记住上一次的值
        M_i_acc = M_i_new
        O_i_acc = O_i_new
        L_i_acc = L_i_new

        # 只有 K 和 V 指针在 K 维度上前进，遍历所有 K/V 分块
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    O_i = O_i_acc / L_i_acc
    L_i = M_i_acc + tl.log(L_i_acc)
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)

@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq_d,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    j = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    
    # 这里offset每个指针，对应batch index乘以batch stride
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),#stride是每个元素存储布局的方式
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),#(1, 0)代表行主序，(0, 1)代表列主序
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq_d, 1),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(j * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )


    # 加载当前K/V tile，这些在整个内层循环中保持不变
    K_j = tl.load(K_block_ptr) #(K_TILE_SIZE, D)
    V_j = tl.load(V_block_ptr) #(K_TILE_SIZE, D)
    
    # 初始化dK和dV的累积器，按照伪代码要求
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    # 内层循环：遍历所有Q tiles
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # 加载当前Q tile的所有相关数据
        Q_i = tl.load(Q_block_ptr) #(Q_TILE_SIZE, D)
        O_i = tl.load(O_block_ptr) #(Q_TILE_SIZE, D)
        dO_i = tl.load(dO_block_ptr) #(Q_TILE_SIZE, D)
        L_i = tl.load(L_block_ptr) #(Q_TILE_SIZE, 1)
        
        # 加载预计算的D_i = rowsum(dO_i ○ O_i)，按照伪代码
        D_i = tl.load(D_block_ptr) #(Q_TILE_SIZE, 1)

        # 计算attention scores S_ij = Q_i @ K_j^T / √d
        S_ij = tl.dot(Q_i, K_j.T) * scale #(Q_TILE_SIZE, K_TILE_SIZE)
        
        # 应用因果掩码
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            causal_mask = q_idx >= k_idx
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        P_ij = tl.exp(S_ij - L_i) #(Q_TILE_SIZE, K_TILE_SIZE)

        dV_j += tl.dot(P_ij.to(V_j.dtype).T, dO_i)
        
        dP_ij = tl.dot(dO_i, V_j.T) #(Q_TILE_SIZE, K_TILE_SIZE)
        
        dS_ij = P_ij * (dP_ij - D_i) * scale #(Q_TILE_SIZE, K_TILE_SIZE)
        dS_ij = dS_ij.to(Q_i.dtype)
        
        # 计算dQ的贡献，使用block指针进行原子累积
        dQ_contribution = tl.dot(dS_ij, K_j) #(Q_TILE_SIZE, D)
        # 生成Q tile和D的索引
        q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
        d_idx = tl.arange(0, D)[None, :]
        # 创建边界掩码，避免对越界的无效的位置进行操作
        q_mask = q_idx < N_QUERIES
        d_mask = d_idx < D
        mask = q_mask & d_mask
        # 计算偏移地址
        offset = batch_index * stride_qb + q_idx * stride_qq + d_idx * stride_qd
        dq_ptrs = dQ_ptr + offset
        # 使用原子操作累积
        tl.atomic_add(dq_ptrs, dQ_contribution.to(dQ_ptr.dtype.element_ty), mask=mask)
        
        # 累积到dK_j
        dK_j += tl.dot(dS_ij.T, Q_i)
        
        # 前进到下一个Q tile
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, 0))
        dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))

    # 写回累积的dK和dV梯度
    tl.store(dK_block_ptr, dK_j.to(dK_ptr.dtype.element_ty))
    tl.store(dV_block_ptr, dV_j.to(dV_ptr.dtype.element_ty))

class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # 处理批处理维度: q, k, v 的形状是 (batch_size, seq_len, d_model)
        batch_size = q.shape[0]
        Nq = q.shape[1] # q 的序列长度
        Nk = k.shape[1] # k 的序列长度
        d = q.shape[2]  # 特征维度
        
        Bq = 32  # q 的分块大小,是一个单位 (节省共享内存)
        Bk = 32  # k 的分块大小
        Tq = Nq // Bq

        scale = 1 / d**0.5

        O = torch.zeros_like(q)
        L = torch.zeros(batch_size, Nq, device=q.device)
        #这里的grid代表启动triton kernel的grid的形状，也就是并行的线程，往往会把batch_size这种常用的并行方式放在后面，但对性能没有影响
        grid = (Tq, batch_size)

        flash_fwd_kernel[grid](q, k, v, O, L,
                               q.stride(0), q.stride(1), q.stride(2),
                               k.stride(0), k.stride(1), k.stride(2),
                               v.stride(0), v.stride(1), v.stride(2),
                               O.stride(0), O.stride(1), O.stride(2),
                               L.stride(0), L.stride(1),
                               Nq, Nk,
                               scale,
                               D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
                               is_causal=is_causal)
        
        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        is_causal = ctx.is_causal

        batch_size = q.shape[0]
        Nq = q.shape[1]
        Nk = k.shape[1]
        d = q.shape[2]

        Bq = 32
        Bk = 32
        # Tq = Nq // Bq
        Tk = Nk // Bk
        scale = 1 / d**0.5

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        D_sum = torch.sum(o * do, dim=-1, keepdim=True)  # 计算D = rowsum(dO ○ O)
        grid = (Tk, batch_size) 
        flash_bwd_kernel[grid](
            q, k, v, o, l, do, D_sum,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            do.stride(0), do.stride(1), do.stride(2),
            D_sum.stride(0), D_sum.stride(1),
            Nq, Nk,
            scale,
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )

        return dq, dk, dv, None