import torch

class Autograd_function_pytorch(torch.autograd.Function):
    """
    Query形状(Nq, d)
    Key形状(Nk, d)
    Value形状(Nk, d)
    """
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
        Tk = Nk // Bk
        
        # 初始化输出张量
        O = torch.zeros_like(q)  # (batch_size, Nq, d)
        L = torch.zeros(batch_size, Nq, device=q.device)  # (batch_size, Nq)
        
        # 对每个批次分别处理
        for b in range(batch_size):
            q_batch = q[b]  # (Nq, d)
            k_batch = k[b]  # (Nk, d) 
            v_batch = v[b]  # (Nk, d)
            
            for i in range(Tq): #外层循环：q 的分块
                max_S_ij_last = torch.tensor([-float('inf')] * Bq, device=q.device).unsqueeze(1) #(Bq, 1)
                l_ij = torch.zeros(Bq, 1, device=q.device) #(Bq, 1)
                O_ij = torch.zeros(Bq, d, device=q.device) #(Bq, d)
                q_i = q_batch[i * Bq:(i+1) * Bq, :]
                for j in range(Tk): #内层循环：k,v 的分块
                    k_j = k_batch[j * Bk:(j+1) * Bk, :]
                    v_j = v_batch[j * Bk:(j+1) * Bk, :]
                    S_ij = q_i @ k_j.T / d**0.5 #(Bq, Bk)
                    max_S_ij_now = torch.max(max_S_ij_last, torch.max(S_ij,dim=1)[0].unsqueeze(1)) #(Bq, 1)
                    #接下来需要S的每一行都减去对应行的最大值
                    P_ij = torch.exp(S_ij - max_S_ij_now) #(Bq, Bk)
                    l_ij = torch.exp(max_S_ij_last - max_S_ij_now) * l_ij + torch.sum(P_ij, dim=1).unsqueeze(1) #(Bq, 1)
                    O_ij = torch.exp(max_S_ij_last - max_S_ij_now) * O_ij + P_ij @ v_j #(Bq, d) 取diag乘对角矩阵其实就是对原矩阵进行缩放，这里是为了动态更新最大值,实际直接广播效率更高
                    #需要维护一个上一个时刻的max_S_ij
                    max_S_ij_last = max_S_ij_now
                #每次内循环结束，l_ij就是大S的分块行的行和
                O_i = torch.diag(1.0/l_ij.squeeze(1)) @ O_ij #(Bq, d)
                L_i = max_S_ij_now.squeeze(1) + torch.log(l_ij.squeeze(1)) #(Bq,)
                
                # 将结果存储到对应的批次和分块位置
                O[b, i * Bq:(i+1) * Bq, :] = O_i
                L[b, i * Bq:(i+1) * Bq] = L_i
        
        # 保存前向传播需要的张量以供反向传播使用
        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        
        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        """
        return:dq,dk,dv
        """
        batch_size, Nq, d = Q.shape
        _, Nk, _ = K.shape
        Bq = 64
        Bk = 64
        Tq = Nq // Bq
        Tk = Nk // Bk

        D = torch.sum(O * dO,dim=-1)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for i in range(Tq):
            #需要按照Bq分块的在第一个循环里面处理
            q_i = Q[:, i * Bq:(i+1) * Bq, :] #(Batch_size, Bq, d)
            dO_i = dO[:, i * Bq:(i+1) * Bq, :] #(Batch_size, Bq, d)
            L_i = L[:, i * Bq:(i+1) * Bq] #(Batch_size, Bq)
            D_i = D[:, i * Bq:(i+1) * Bq] #(Batch_size, Bq)

            for j in range(Tk):
                #需要按照Bk分块的在第二个循环里面处理
                k_j = K[:, j * Bk:(j+1) * Bk, :]
                v_j = V[:, j * Bk:(j+1) * Bk, :]

                S_ij = q_i @ k_j.transpose(-2,-1) / (d**0.5)
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))

                dV_j = P_ij.transpose(-2,-1) @ dO_i
                dV[:, j * Bk:(j+1) * Bk, :] += dV_j #这里要注意dV、dK、dQ都是需要累计梯度的，因为V的第 j 个分块 V_j，在前向传播中，它不仅和 Q 的第0个分块 Q_0 作用了，也和 Q_1, Q_2, ... Q_{Tq-1} 都发生了作用。
                
                dP_ij = dO_i @ v_j.transpose(-2,-1)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))
                
                dQ_i_j = (dS_ij @ k_j) / (d**0.5)
                dQ[:, i * Bq:(i+1) * Bq, :] += dQ_i_j
                
                dK_j = (dS_ij.transpose(-2,-1) @ q_i) / (d**0.5)
                dK[:, j * Bk:(j+1) * Bk, :] += dK_j

        return dQ,dK,dV,None
