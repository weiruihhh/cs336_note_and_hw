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
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    i = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    
    # 这里offset每个指针，对应batch index乘以batch stride
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),#stride是每个元素存储布局的方式
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),#(1, 0)代表行主序，(0, 1)代表列主序
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
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
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
        offsets=(0, 0),
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
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_i = tl.load(Q_block_ptr) #(Q_TILE_SIZE, D)
    O_i = tl.load(O_block_ptr) #(Q_TILE_SIZE, D)
    dO_i = tl.load(dO_block_ptr) #(Q_TILE_SIZE, D)
    L_i = tl.load(L_block_ptr) #(Q_TILE_SIZE, 1)

    D_i = tl.sum(O_i * dO_i, axis=1, keep_dims=True)
    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr) #(K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr) #(K_TILE_SIZE, D)

        S_ij = tl.dot(Q_i, K_j.T) * scale #(Q_TILE_SIZE, K_TILE_SIZE)
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            causal_mask = q_idx >= k_idx
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        P_ij = tl.exp(S_ij - L_i) #(Q_TILE_SIZE, K_TILE_SIZE)
        

        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i) * scale#这里的P跟S还都是和V、O、D一样的类型，接下来转换成和K、Q一样的类型
        dS_ij = dS_ij.to(Q_i.dtype)

        
        dQ_i += tl.dot(dS_ij, K_j)        
        dK_j = tl.dot(dS_ij.T, Q_i)
        dV_j = tl.dot(P_ij.to(V_j.dtype).T, dO_i)

        # tl.atomic_add(dK_block_ptr, dK_j.to(dK_ptr.dtype.element_ty))#原子操作，用于多个线程同时访问同一个地址时，保证数据的一致性
        # tl.atomic_add(dV_block_ptr, dV_j.to(dV_ptr.dtype.element_ty))
        tl.atomic_add(dK_block_ptr, dK_j)
        tl.atomic_add(dV_block_ptr, dV_j)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        dK_block_ptr = dK_block_ptr.advance((K_TILE_SIZE, 0))
        dV_block_ptr = dV_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_i.to(Q_ptr.dtype.element_ty))


class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # 处理批处理维度: q, k, v 的形状是 (batch_size, seq_len, d_model)
        batch_size = q.shape[0]
        Nq = q.shape[1] # q 的序列长度
        Nk = k.shape[1] # k 的序列长度
        d = q.shape[2]  # 特征维度
        
        Bq = 64  # q 的分块大小,是一个单位
        Bk = 64  # k 的分块大小
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

        Bq = 64
        Bk = 64
        Tq = Nq // Bq

        scale = 1 / d**0.5

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        grid = (Tq, batch_size)
        flash_bwd_kernel[grid](
            q, k, v, o, l, do,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            do.stride(0), do.stride(1), do.stride(2),
            Nq, Nk,
            scale,
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )

        return dq, dk, dv, None