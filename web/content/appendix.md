---
outline: [2, 3]
---

# 基础知识速查

按需查阅：[数学基础](#app-math) · [深度学习基础](#app-deep-learning) · [AI Infra 基础](#app-infra) · [信息论基础](#app-information) · [正则表达式](#app-regex)。

<span id="app-repro"></span>

::: info 实验复现与结果记录
新版附录已按知识领域重新组织。前文的旧链接保留在此：训练日志与 checkpoint 见[训练流程与实验管理](/part-1/chapter-4)，计时范围与环境记录见[性能分析与基准测试](/part-2/chapter-6)。
:::

## 数学基础

<span id="app-math"></span> 本附录按正文中的使用顺序补足三组知识：先用矩阵微分说明梯度、反向传播和局部曲率，再用概率论解释训练目标与采样，最后用约束优化连接信赖域和偏好对齐。公式中的向量默认是实列向量，未注明底数的 $\log$ 均指自然对数。不同章节可能复用字母，查阅时以局部定义为准。

<strong class="note-label">怎样配合正文阅读：</strong>阅读[Transformer 架构](/part-1/chapter-2#guide-ch-2)和[语言模型的训练](/part-1/chapter-3#guide-ch-3)时，可先查矩阵求导、链式法则与最大似然；阅读[文本生成与解码](/part-1/chapter-5#guide-ch-5)时，可查分布与采样；阅读[Scaling Law](/part-3/chapter-9#guide-ch-10)时，可查幂律关系；阅读[策略梯度与 PPO](/part-5/chapter-12#guide-ch-12-policy)、[GRPO 与 TRPO 补充推导](/part-5/chapter-13#guide-ch-12-grpo)及[偏好对齐理论](/part-6/chapter-14#guide-ch-13)时，可查重要性采样、Fisher 信息、拉格朗日对偶与 KKT 条件。熵的基础定义另见[信息论基础](/appendix#app-information)。

### 线性代数与矩阵求导

<span id="app-math-matrix"></span>

#### 对象、维度与内积：先确定对什么求导

标量是单个数，向量是一列数，矩阵是二维数组，更高维数组在实现中通常称为张量。若 $A\in\mathbb R^{m\times k}$、$B\in\mathbb R^{k\times n}$，则 $AB\in\mathbb R^{m\times n}$；$A^\top$ 交换行列，$A\odot B$ 则表示形状匹配时的逐元素乘法。两种乘法的求导规则不能混用。

例如 $X\in\mathbb R^{B\times T\times d}$、$W\in\mathbb R^{d\times h}$ 时，线性层在最后一维上计算，输出 $XW\in\mathbb R^{B\times T\times h}$。推导时可先把前两维合并成 $N=BT$，按 $X\in\mathbb R^{N\times d}$ 处理；实现时再还原批量与序列轴。

向量内积为 $a^\top b$。矩阵的对应概念是 <strong class="key-term">Frobenius 内积</strong>：

$$
\langle A,B\rangle_F=\operatorname{tr}(A^\top B)
=\sum_{i,j}A_{ij}B_{ij},\qquad
\|A\|_F^2=\langle A,A\rangle_F.
$$

这里 $\operatorname{tr}(M)=\sum_i M_{ii}$ 是方阵的迹。形状允许时，迹可以循环移动因子：

$$
\operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB).
$$

<strong class="critical-term">循环移动不等于任意交换。</strong>一般不能把 $ABC$ 改成 $ACB$。迹的作用是把标量表达式中的微分项移动到便于识别梯度的位置。

#### 微分、梯度与 Jacobian 的约定

<span id="app-math-differential"></span> 对标量函数 $f:\mathbb R^n\to\mathbb R$，本附录采用列梯度：

$$
\nabla_x f=
\begin{bmatrix}\partial f/\partial x_1&\cdots&\partial f/\partial x_n\end{bmatrix}^{\!\top},
\qquad df=(\nabla_x f)^\top dx.
$$

这个等式给出了求梯度的操作方法：先展开 $df$，再整理成“一个列向量的转置乘 $dx$”，该列向量就是梯度。对矩阵参数 $X\in\mathbb R^{m\times n}$，相同约定写成

<div class="key-formula">

$$
df=\langle\nabla_X f,dX\rangle_F
=\operatorname{tr}\bigl((\nabla_X f)^\top dX\bigr),
\qquad (\nabla_X f)_{ij}=\frac{\partial f}{\partial X_{ij}}.
$$

</div>

因此标量损失对矩阵的梯度与矩阵形状相同。不同资料可能采用不同的导数排布；比较公式时，先确认约定，再检查维度。

若 $f:\mathbb R^n\to\mathbb R^m$ 是向量函数，其一阶导数是 <strong class="key-term">Jacobian（雅可比矩阵）</strong>：

$$
J_f(x)=\begin{bmatrix}
\frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_n}\\
\vdots&\ddots&\vdots\\
\frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}
\end{bmatrix}\in\mathbb R^{m\times n},\qquad df=J_f(x)\,dx.
$$

每一行对应一个输出，每一列对应一个输入。标量输出时，$J_f=(\nabla f)^\top$；Jacobian 与梯度相差的转置来自这一定义。

#### 矩阵微分的基本规则

微分对加减法、乘法和转置的规则为

$$
\begin{aligned}
d(X\pm Y)&=dX\pm dY,\\
d(XY)&=(dX)Y+X(dY),\\
d(X^\top)&=(dX)^\top,\\
d\operatorname{tr}(X)&=\operatorname{tr}(dX).
\end{aligned}
$$

矩阵乘法通常不可交换，所以乘法法则必须保留因子顺序。若方阵 $X$ 可逆，由 $XX^{-1}=I$ 两边微分得到

$$
(dX)X^{-1}+X\,d(X^{-1})=0
\quad\Longrightarrow\quad
d(X^{-1})=-X^{-1}(dX)X^{-1}.
$$

行列式的微分为

$$
d\det X=\operatorname{tr}\bigl(\operatorname{adj}(X)dX\bigr).
$$

其中 $\operatorname{adj}(X)$ 是伴随矩阵，该形式也适用于奇异矩阵。若 $X$ 可逆，可进一步写成

$$
d\det X=\det X\operatorname{tr}(X^{-1}dX).
$$

若再满足 $\det X>0$，则实数域中的 $\log\det X$ 有定义，且

$$
d\log\det X=\operatorname{tr}(X^{-1}dX),\qquad
\nabla_X\log\det X=X^{-\top}.
$$

正定矩阵是经常使用这些公式的情形。公式中的可逆性、定义域条件都是推导的一部分，不能省略后直接代入任意矩阵。

#### 常用梯度及两种推导示范

<span id="app-math-derivative-formulas"></span>

###### 向量变量

设 $x,a,b\in\mathbb R^n$，$A\in\mathbb R^{n\times n}$，且 $a,b,A$ 不依赖于 $x$，则

$$
\begin{aligned}
\nabla_x(x^\top a)=\nabla_x(a^\top x)&=a,\\
\nabla_x(x^\top x)&=2x,\\
\nabla_x(x^\top Ax)&=(A+A^\top)x,\\
\nabla_x(a^\top xx^\top b)&=ab^\top x+ba^\top x.
\end{aligned}
$$

以二次型为例，乘法法则给出

$$
\begin{aligned}
d(x^\top Ax)
&=(dx)^\top Ax+x^\top A\,dx\\
&=(Ax)^\top dx+(A^\top x)^\top dx\\
&=\bigl((A+A^\top)x\bigr)^\top dx.
\end{aligned}
$$

所以只有 $A$ 对称时，梯度才可直接写成 $2Ax$。最后一个公式也可以将目标写成 $(a^\top x)(b^\top x)$，对两个标量因子应用乘法法则。

###### 矩阵变量

下面每个公式中的 $a,b$ 都是常向量，$X\in\mathbb R^{m\times n}$。向量维度随所在公式明确指定：

$$
\begin{aligned}
a\in\mathbb R^m,\ b\in\mathbb R^n:
&\quad\nabla_X(a^\top Xb)=ab^\top;\\
a\in\mathbb R^n,\ b\in\mathbb R^m:
&\quad\nabla_X(a^\top X^\top b)=ba^\top;\\
a,b\in\mathbb R^m:
&\quad\nabla_X(a^\top XX^\top b)=(ab^\top+ba^\top)X;\\
a,b\in\mathbb R^n:
&\quad\nabla_X(a^\top X^\top Xb)=X(ba^\top+ab^\top).
\end{aligned}
$$

例如第一式用迹整理为

$$
d(a^\top Xb)=a^\top(dX)b
=\operatorname{tr}(ba^\top dX)
=\operatorname{tr}\bigl((ab^\top)^\top dX\bigr).
$$

对照 Frobenius 梯度定义即可读出 $ab^\top$，形状恰为 $m\times n$。这比凭外形记忆转置的位置更可靠。

###### 逐元素非线性

令 $z=Xb\in\mathbb R^m$、$a\in\mathbb R^m$。若 $e^z$ 指对每个元素取指数，则

$$
f(X)=a^\top e^{Xb},\qquad
\nabla_X f=(a\odot e^{Xb})b^\top.
$$

先求 $\nabla_z f=a\odot e^z$，再通过 $z=Xb$ 传回 $X$。这里的指数是<strong class="key-term">逐元素指数</strong>，不是方阵的矩阵指数 $\exp(X)=\sum_{k\geq0}X^k/k!$。

#### 链式法则为什么会出现转置

<span id="app-math-chain-rule"></span> 设 $y=f(x)$，标量损失 $L=\ell(y)$。由

$$
dL=(\nabla_y L)^\top dy
=(\nabla_y L)^\top J_f\,dx
$$

可得

<div class="key-formula">

$$
\nabla_x L=J_f(x)^\top\nabla_y L.
$$

</div>

正向传播把输入扰动 $dx$ 乘以 $J_f$ 得到输出扰动；反向传播则把输出端梯度乘以 $J_f^\top$ 传回输入端。实际系统通常直接计算这一向量乘积，不必显式构造完整 Jacobian。

对于 $Y=AX$，设 $A\in\mathbb R^{m\times n}$ 固定，$X\in\mathbb R^{n\times k}$，上游梯度 $G=\nabla_Y L\in\mathbb R^{m\times k}$。由

$$
dL=\operatorname{tr}(G^\top A\,dX)
=\operatorname{tr}\bigl((A^\top G)^\top dX\bigr)
$$

得到 $\nabla_X L=A^\top G$。$k=1$ 就是列向量的情形。

正文线性层常写为 $Y=XW$。此时 $X\in\mathbb R^{N\times d}$、$W\in\mathbb R^{d\times h}$、$G\in\mathbb R^{N\times h}$，同时考虑两种参数变化：

$$
dY=(dX)W+X(dW),\qquad
\boxed{\nabla_X L=GW^\top,\quad\nabla_W L=X^\top G.}
$$

每条梯度都应回到对应输入的形状。若加入按行广播的偏置 $b\in\mathbb R^h$，则偏置梯度为 $\nabla_bL=\sum_{i=1}^N G_{i,:}^{\top}$；这也解释了广播操作在反向传播时为什么需要求和。

###### 平方误差：把一阶、二阶导数串起来

设 $X\in\mathbb R^{N\times d}$ 固定，$w\in\mathbb R^d$，$y\in\mathbb R^N$，令 $r=Xw-y$。则

$$
L(w)=\|r\|_2^2,\quad dL=2r^\top X\,dw,\quad
\nabla_wL=2X^\top(Xw-y),\quad H_L=2X^\top X.
$$

若损失前有 $1/2$，两个导数中的因子 2 消失；若按样本取平均，还需除以 $N$。求和和平均并非排版区别，它们会改变实际更新的尺度。

###### Softmax 与交叉熵的梯度

设 logits 为 $z\in\mathbb R^V$，$p_i=e^{z_i}/\sum_j e^{z_j}$，则

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j),\qquad
J_{\mathrm{softmax}}=\operatorname{diag}(p)-pp^\top.
$$

这里 $\delta_{ij}$ 在 $i=j$ 时为 1，否则为 0。对满足 $y_i\geq0$、$\sum_i y_i=1$ 的目标分布，

$$
L=-\sum_i y_i\log p_i
=-y^\top z+\log\sum_j e^{z_j}
\quad\Longrightarrow\quad\nabla_z L=p-y.
$$

当 $y$ 是真实 token 的 one-hot 向量时，这就是语言模型单个预测位置的梯度。它对 logits 的 Hessian 是 $\operatorname{diag}(p)-pp^\top\succeq0$，但 logits 是网络参数的非线性函数，不能据此推断整个神经网络的损失对参数也凸。

#### Taylor 展开：用线性项和二次项近似函数

<span id="app-math-taylor"></span>

###### 向量函数的一阶展开

若 $f:\mathbb R^n\to\mathbb R^m$ 在 $x_0$ 可微，记 $\Delta=x-x_0$，则

<div class="key-formula">

$$
f(x_0+\Delta)=f(x_0)+J_f(x_0)\Delta+o(\|\Delta\|).
$$

</div>

小 $o$ 表示余项的范数除以 $\|\Delta\|$ 后趋于零。忽略余项，就得到常用的一阶近似。Jacobian 的形状是 $m\times n$，乘上 $n$ 维扰动后得到 $m$ 维输出变化。

例如 $f(x_1,x_2)=(x_1^2,x_1x_2)^\top$，在 $(1,2)^\top$ 处

$$
f(x_0)=\begin{bmatrix}1\\2\end{bmatrix},\quad
J_f(x_0)=\begin{bmatrix}2&0\\2&1\end{bmatrix},\quad
f(x_0+\Delta)\approx
\begin{bmatrix}1+2\Delta_1\\2+2\Delta_1+\Delta_2\end{bmatrix}.
$$

被忽略的部分恰为 $(\Delta_1^2,\Delta_1\Delta_2)^\top$，因此扰动足够小时，一阶项主导变化。

###### 标量函数的二阶展开与 Hessian

对邻域内二阶连续可导的标量函数 $f:\mathbb R^n\to\mathbb R$，定义 <strong class="key-term">Hessian（海森矩阵）</strong>

$$
H_f(x)=\left[\frac{\partial^2 f}{\partial x_i\partial x_j}\right]_{i,j=1}^n.
$$

对角线是纯二阶偏导，非对角线是混合偏导；在上述连续性条件下，混合偏导可交换，故 $H_f=H_f^\top$。二阶展开为

<div class="key-formula">

$$
\begin{aligned}
f(x_0+\Delta)={}&f(x_0)+\nabla f(x_0)^\top\Delta\\
&+\frac12\Delta^\top H_f(x_0)\Delta+o(\|\Delta\|^2).
\end{aligned}
$$

</div>

梯度给出一阶变化；$v^\top H_fv$ 描述沿方向 $v$ 的二阶变化。令 $\phi(t)=f(x_0+tv)$，便有 $\phi'(0)=\nabla f(x_0)^\top v$、$\phi''(0)=v^\top H_f(x_0)v$。所谓“曲率”在这里指沿参数方向的二阶变化，不必等同于曲面几何中的所有曲率定义。

向量函数的二阶项需要对每个输出 $f_i$ 分别使用 Hessian：第 $i$ 个分量的二阶项为 $\tfrac12\Delta^\top H_{f_i}\Delta$。一般不存在一个普通的 $m\times n$ Hessian 矩阵能像 Jacobian 那样包办所有二阶导数。

#### 正定性、特征值与极值判定

<span id="app-math-positive-definite"></span> 本附录对实对称矩阵 $A=A^\top$ 讨论正定性。二次型 $v^\top Av$ 是一个标量，定义如下：

$$
\begin{array}{ll}
A\succ0\text{（正定）}:&v^\top Av>0\quad\text{对所有 }v\ne0;\\
A\succeq0\text{（半正定）}:&v^\top Av\geq0\quad\text{对所有 }v;\\
A\prec0\text{（负定）}:&v^\top Av<0\quad\text{对所有 }v\ne0.
\end{array}
$$

负半定同理；若二次型在不同方向上能分别取正值和负值，则称为不定。这里的 $A\succeq0$ 是二次型意义下的比较，<strong class="critical-term">不是逐元素非负</strong>。

由实对称矩阵的谱分解 $A=Q\Lambda Q^\top$，其中 $Q^\top Q=I$，令 $z=Q^\top v$，可得

$$
v^\top Av=z^\top\Lambda z=\sum_i\lambda_i z_i^2.
$$

所以所有特征值严格为正等价于正定，所有特征值非负等价于半正定，既有正特征值又有负特征值则是不定。对二次函数 $\tfrac12v^\top Av$，正定对应各个主方向都向上弯的“碗”，负定对应“山峰”，不定对应“马鞍”。半正定允许某些方向是平的。

###### Hessian 能说明什么，不能说明什么

设 $x_*$ 是无约束问题的内点，$f$ 在其邻域二阶连续可导。

- 若 $x_*$ 是局部极小值，则必要条件是 $\nabla f(x_*)=0$ 且 $H_f(x_*)\succeq0$。

- 若 $\nabla f(x_*)=0$ 且 $H_f(x_*)\succ0$，则 $x_*$ 是严格局部极小值；负定时是严格局部极大值。

- 在驻点处，Hessian 不定意味着存在上升与下降方向，是鞍点。

- Hessian 半正定但不正定时，单靠二阶信息通常无法判定，必须看更高阶项或其他结构。

例如 $f(x)=x^4$ 在 0 处有严格最小值，但 $f''(0)=0$；$-x^4$ 的二阶导数同样为零，却有严格最大值。再如 $f(x,y)=x^2-y^4$ 在原点 Hessian 为 $\operatorname{diag}(2,0)\succeq0$，沿 $x$ 轴上升、沿 $y$ 轴下降，仍是鞍点。因而不能把“梯度为零且 Hessian 正定”说成所有极小值必须同时满足的条件。

边界最优点还可能有非零梯度，例如 $\min_{x\geq0}x$ 在 $x=0$ 最优。约束限制了可走的方向，此时应使用后面的约束最优性条件。

###### 贯穿例子：$f(x,y)=x^2+xy+y^2$

$$
\nabla f=\begin{bmatrix}2x+y\\x+2y\end{bmatrix},\qquad
H_f=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$

令梯度为零得 $(x,y)=(0,0)$。Hessian 的特征值为 1 和 3，处处正定，因此函数严格凸，原点是唯一的全局最小值。也可直接配方：

$$
f(x,y)=\left(x+\frac y2\right)^2+\frac34y^2\geq0,
$$

等号仅在原点成立。这里能得到全局结论，是因为掌握了整个函数的结构，而不仅是在某一点看到了正定 Hessian。

#### 从曲率到数值优化：Hessian、SVD 与矩阵向量积

用二阶模型近似损失 $L(\theta+\Delta)$，对 $\Delta$ 求导并令其为零，得到牛顿方程

$$
H_L(\theta)\Delta=-\nabla L(\theta).
$$

当 Hessian 正定时，这个二次模型有唯一最小值；若 Hessian 不定或奇异，直接使用牛顿方向未必下降，通常需要阻尼、信赖域或其他修正。BFGS、L-BFGS 等拟牛顿方法利用迭代中的参数与梯度变化近似曲率，其中 L-BFGS 只保存有限的历史向量。

大型模型不会轻易显式存储 $n\times n$ 的 Hessian。对固定向量 $v$，可以通过

$$
H_Lv=\nabla_\theta\bigl(\nabla_\theta L(\theta)^\top v\bigr)
$$

计算 Hessian 向量积，并结合迭代线性求解器。这里求导时将 $v$ 视为常量。公式中出现 $H^{-1}g$，通常应理解为求解 $Hv=g$，不意味着实现中必须计算逆矩阵。

另一个常见工具是奇异值分解（SVD）：任意实矩阵 $X$ 可写为 $X=U\Sigma V^\top$，奇异值 $\sigma_i\geq0$。它们与 $X^\top X$ 的非零特征值满足 $\lambda_i=\sigma_i^2$，并给出

$$
\|X\|_F^2=\sum_i\sigma_i^2,\qquad
\|X\|_2=\max_i\sigma_i.
$$

第一式是 Frobenius 范数，第二式是谱范数。$X^\top X$ 总是半正定，因为 $v^\top X^\top Xv=\|Xv\|_2^2\geq0$；只有 $X$ 列满秩时它才正定。正文中的低秩近似、矩阵更新和 Muon 正交化可据此理解；特征值与奇异值的定义不同，不能对一般矩阵混用。

### 概率论基础

<span id="app-math-probability"></span>

#### 概率质量、概率密度与累积分布

<span id="app-math-distribution"></span> 随机变量把随机试验的结果映射为数值。对离散随机变量 $X$，概率质量函数（PMF）$p(x)=\Pr(X=x)$ 满足 $p(x)\geq0$、$\sum_xp(x)=1$。语言模型在有限词表上输出的就是这样的离散分布。

对具有密度的连续随机变量，概率密度函数（PDF）满足

$$
p(x)\geq0,\qquad\int_{-\infty}^{\infty}p(x)\,dx=1,\qquad
\Pr(a\leq X\leq b)=\int_a^b p(x)\,dx.
$$

<strong class="critical-term">密度值本身不是概率。</strong>例如 $X\sim\operatorname{Uniform}(0,\tfrac12)$ 时，区间内 $p(x)=2$，但总面积仍是 1，$\Pr(0\leq X\leq\tfrac14)=\tfrac12$。有密度的连续分布在单点上的概率为零，并不意味着该点附近不可能采到样本。

累积分布函数（CDF）对离散和连续随机变量都适用：

$$
F(x)=\Pr(X\leq x).
$$

它单调不减、右连续，取值在 $[0,1]$ 内，并在两端分别趋于 0 和 1。有密度时 $F(x)=\int_{-\infty}^xp(t)\,dt$，在可微处 $F'(x)=p(x)$。离散分布的 CDF 是阶梯状的，所以不能总是假定 CDF 严格递增或存在普通反函数。

#### 条件概率、链式分解与独立性

离散情形中，若 $\Pr(X=x)>0$，条件概率定义为

$$
p(y\mid x)=\frac{p(x,y)}{p(x)},\qquad p(x,y)=p(x)p(y\mid x).
$$

连续情形可在适当条件下用相应密度的比值定义条件密度。将乘法规则重复使用，可得

<div class="key-formula">

$$
p(x_{1:T})=\prod_{t=1}^T p(x_t\mid x_{<t}),\qquad
\log p(x_{1:T})=\sum_{t=1}^T\log p(x_t\mid x_{<t}).
$$

</div>

这正是自回归语言模型的分解。它来自条件概率的链式法则，<strong class="key-term">不要求 token 相互独立</strong>；相反，条件中的前缀就是模型利用上下文依赖的方式。

若随机变量独立，联合概率才可分解成各自边缘概率的乘积；“同分布”则表示它们服从相同分布。独立同分布（i.i.d.）是两项假设，不是样本数量多时自然成立的事实。训练数据可被建模为独立抽取的序列，但一条序列内部的 token 仍通常相关。

#### 期望、方差与用样本估计平均量

期望是按分布加权的平均：

$$
\mathbb E_{X\sim p}[h(X)]=
\begin{cases}
\sum_xp(x)h(x),&\text{离散情形},\\
\int p(x)h(x)\,dx,&\text{有密度的连续情形}.
\end{cases}
$$

对独立同分布样本 $X_1,\ldots,X_N\sim p$，用

$$
\widehat\mu=\frac1N\sum_{i=1}^Nh(X_i)
$$

估计 $\mu=\mathbb E_p[h(X)]$。在 $\mathbb E_p|h(X)|<\infty$ 等通常的大数定律条件下，样本均值随样本增多趋近于期望。若方差 $\sigma_h^2<\infty$，则 $\mathbb E[\widehat\mu]=\mu$，且

$$
\operatorname{Var}(\widehat\mu)=\frac{\sigma_h^2}{N},\qquad
\operatorname{Var}(Z)=\mathbb E[(Z-\mathbb EZ)^2].
$$

标准差是方差的平方根。上述 $1/N$ 方差缩减依赖独立性；若样本相关，方差中还会出现协方差项。这是阅读批量梯度、回报均值与优势估计时需要注意的区别。

正文中的损失归约还涉及<strong class="key-term">平均的单位</strong>。令第 $i$ 条序列有 $T_i$ 个有效 token，则

$$
\frac1N\sum_i\frac1{T_i}\sum_t\ell_{i,t}
\quad\text{与}\quad
\frac{\sum_i\sum_t\ell_{i,t}}{\sum_iT_i}
$$

一般不同：前者每条序列等权，后者每个有效 token 等权。计算时还应排除 padding 或被 mask 的位置。

###### 重要性采样：样本来自另一个分布时怎么办

若目标是 $\mathbb E_p[h(X)]$，但样本来自 $q$，且 $p$ 有质量的地方 $q$ 也有质量，则

$$
\mathbb E_p[h(X)]
=\mathbb E_q\left[\frac{p(X)}{q(X)}h(X)\right].
$$

以离散情形为例，只需把 $\sum_xp(x)h(x)$ 写成 $\sum_xq(x)\frac{p(x)}{q(x)}h(x)$。这个支持集条件很重要：若某些目标事件永远不会被 $q$ 采到，权重无法凭空补回它们。

在固定上下文或状态 $s$ 下，旧策略采样的动作可用 $\pi_\theta(a\mid s)/\pi_{\mathrm{old}}(a\mid s)$ 重加权，这就是策略优化中概率比的来源。比值很大时估计方差可能很高。PPO 的裁剪改变了优化目标，不能再简单当成原期望的无偏重写；此外，只重加权动作并不会自动修正状态访问分布的变化。

#### 概率与似然：同一个表达式，不同的自变量

<span id="app-math-likelihood"></span> 写出参数化模型 $p_\theta(x)$ 后，有两种观察方式：

- <strong class="list-label">概率或密度：</strong>固定参数 $\theta$，让观测值 $x$ 变化，描述模型会产生怎样的数据。

- <strong class="list-label">似然：</strong>固定已经观察到的数据 $x$，把同一个数值表达式看作参数的函数，记为 $L(\theta;x)=p_\theta(x)$。

区别是<strong class="key-term">谁固定、谁变化</strong>，不是“事件发生前才有概率，发生后才有似然”。对于连续数据，似然使用的是概率密度，也可以大于 1。似然不必对参数积分为 1，<strong class="critical-term">不是参数的概率分布或后验概率</strong>。

若需要参数的后验分布，还必须给定先验并使用贝叶斯公式：

$$
p(\theta\mid x)=\frac{p(x\mid\theta)p(\theta)}{p(x)},\qquad
p(x)=\int p(x\mid\theta)p(\theta)\,d\theta.
$$

因此 $L(\theta;x)=p(x\mid\theta)$ 是同一模型表达式的重新看待方式，而不是 $p(\theta\mid x)=p(x\mid\theta)$。

#### 最大似然估计：从乘积到训练损失

<span id="app-math-mle"></span> 设数据 $D=\{x_1,\ldots,x_N\}$ 在给定参数后独立同分布，则

$$
L(\theta;D)=\prod_{i=1}^Np_\theta(x_i),\qquad
\ell(\theta)=\log L(\theta;D)=\sum_{i=1}^N\log p_\theta(x_i).
$$

最大似然估计（MLE）为

$$
\hat\theta\in\operatorname*{arg\,max}_{\theta\in\Theta}\ell(\theta).
$$

对数严格递增，不改变最大值的位置；它把很小概率的连乘改成求和，既便于求导，也有助于避免下溢。实现中应直接计算稳定的对数概率，例如使用 log-softmax，而不是先把小概率乘完再取对数。

常用求解流程是：写出模型与参数范围，构造似然，取对数，求内部驻点，再检查边界和最优性。在可微的内部最优点应满足 $\nabla_\theta\ell(\theta)=0$，<strong class="critical-term">不是令 $\ell(\theta)=0$</strong>。驻点可能不是最大值；有些模型的最优解落在边界、并不唯一，甚至不存在有限的最大似然估计。

###### 例一：8000 次正面、2000 次反面

设每次抛硬币独立，正面概率为 $\theta\in[0,1]$，记 $k=8000$、$N=10000$。若记录的是完整有序序列，则

$$
L(\theta)=\theta^k(1-\theta)^{N-k},\qquad
\ell(\theta)=k\log\theta+(N-k)\log(1-\theta).
$$

若只记录正面次数，似然多一个与 $\theta$ 无关的组合系数 $\binom Nk$，最大值位置相同。对内部参数求导：

$$
\ell'(\theta)=\frac{k}{\theta}-\frac{N-k}{1-\theta}=0
\quad\Longrightarrow\quad\hat\theta=\frac{k}{N}=0.8.
$$

当 $0<k<N$ 时，

$$
\ell''(\theta)=-\frac{k}{\theta^2}-\frac{N-k}{(1-\theta)^2}<0,
$$

故该驻点是唯一最大值。若全部为正面或全部为反面，则最优解分别在边界 1 或 0，不能靠内部求导公式寻找驻点。

###### 例二：高斯均值、方差与平方误差

设 $x_i\sim\mathcal N(\mu,\sigma^2)$ 独立，$\sigma>0$ 是标准差，$\sigma^2$ 是方差。对数似然为

$$
\ell(\mu,\sigma^2)=-\frac N2\log(2\pi\sigma^2)
-\frac1{2\sigma^2}\sum_i(x_i-\mu)^2.
$$

对均值求导得到 $\hat\mu=\bar x=\frac1N\sum_ix_i$；令 $v=\sigma^2$ 并对 $v$ 求导，有

$$
-\frac N{2v}+\frac1{2v^2}\sum_i(x_i-\mu)^2=0,
\quad\Longrightarrow\quad
\widehat{\sigma^2}_{\mathrm{MLE}}=\frac1N\sum_i(x_i-\bar x)^2.
$$

这里分母为 $N$；常见的无偏样本方差使用 $N-1$，两者目的不同。若样本全相同，方差趋于 0 时似然无界，在要求 $\sigma^2>0$ 的参数空间内没有有限的最大值。固定方差时，最大化高斯似然等价于最小化平方误差，这解释了平方损失的一种概率来源。

###### 语言模型的负对数似然

令第 $i$ 条训练序列为 $x^{(i)}_{1:T_i}$，根据条件概率链式分解，训练目标可写为

<div class="key-formula">

$$
\mathcal L_{\mathrm{NLL}}(\theta)
=-\sum_i\sum_{t=1}^{T_i}\log p_\theta(x^{(i)}_t\mid x^{(i)}_{<t}).
$$

</div>

每个位置把真实 token 写为 one-hot 分布后，其负对数概率就是该位置的交叉熵。最大似然、最小化 NLL 和这一监督目标下最小化交叉熵，是同一个优化方向的三种表述；是否按序列或 token 平均，需要另外说明。

#### 逆变换采样：把均匀数变成目标分布

<span id="app-math-inverse-sampling"></span> 采样是根据指定分布产生随机取值。许多采样构造以 $U\sim\operatorname{Uniform}(0,1)$ 为起点，再通过变换得到目标分布；实际计算机使用有限精度的伪随机数，并不要求所有分布都只能用这一种算法生成。

###### 从百分位理解 CDF 的反向使用

若 $F(h)$ 表示身高不超过 $h$ 的人所占比例，$F(170)=0.5$ 表示 170 厘米是一个中位数，并不自动等于平均身高。假设 $F(190)=0.9$，那么从百分位 $u=0.9$ 查回身高，可得到 190 厘米。

均匀抽取百分位时，人群密集的身高区间对应较长的百分位区间，所以被抽中的机会也更大。这里使用的是从百分位返回取值的 $F^{-1}$，不是把均匀数代入 $F$。

当 $F$ 连续且严格递增时，可定义 $X=F^{-1}(U)$，于是

$$
\Pr(X\leq x)=\Pr\bigl(F^{-1}(U)\leq x\bigr)
=\Pr(U\leq F(x))=F(x).
$$

这说明所得随机变量的 CDF 正是目标 $F$。一般情形应使用<strong class="key-term">广义逆（分位数函数）</strong>

$$
Q(u)=\inf\{x:F(x)\geq u\},\qquad0<u<1.
$$

即使 CDF 有跳跃或平坦区间，$Q(U)$ 仍服从目标分布。端点及有限精度的处理应遵循具体采样实现。

###### 离散例子：从词表概率到 token

假设三个候选 token 的概率是 $(0.2,0.5,0.3)$，累积概率为 $(0.2,0.7,1)$。把单位区间划分为

$$
(0,0.2]\longrightarrow\text{token 1},\quad
(0.2,0.7]\longrightarrow\text{token 2},\quad
(0.7,1)\longrightarrow\text{token 3}.
$$

均匀数落在哪一段，就选哪一个 token，各段长度正好是对应概率。连续均匀分布下端点的概率为零。正文中的温度、top-$k$、top-$p$ 会先改变候选分布或截断支持集；对保留候选重新归一化后，才按新分布采样。

###### 连续例子：指数分布

若事件按速率为 $\lambda>0$ 的齐次 Poisson 过程到达，则从固定时刻到下一事件的等待时间服从指数分布。把它解释为公交等待时间，需要这种到达假设；固定时刻表的公交不自动满足它。

目标密度和 CDF 为

$$
p(x)=\begin{cases}\lambda e^{-\lambda x},&x\geq0,\\0,&x<0,\end{cases}
\qquad
F(x)=1-e^{-\lambda x}\quad(x\geq0).
$$

解 $u=1-e^{-\lambda x}$ 得到

<div class="key-formula">

$$
X=-\frac1\lambda\log(1-U),\qquad U\sim\operatorname{Uniform}(0,1).
$$

</div>

例如 $\lambda=0.2\,\text{分钟}^{-1}$、$u=0.3$ 时，$x=-5\log0.7\approx1.78$ 分钟。一次抽样只产生一个等待时间，并不等于均值 $1/\lambda=5$ 分钟。实现中可用 `log1p(-u)` 稳定计算 $\log(1-u)$，同时避免把 $u=1$ 代入产生无穷值。

#### 幂律关系与 Scaling Law

<span id="app-math-power-law"></span> 正文中的 Scaling Law 常用幂律描述资源与损失之间的经验关系。对 $x>0$，一个基本形式是

$$
y=kx^{-\alpha},\qquad k>0,\quad\alpha>0.
$$

这表示幂次衰减，而非指数衰减 $ke^{-\alpha x}$。对两边取对数，

$$
\log y=\log k-\alpha\log x.
$$

所以幂律在双对数坐标上是一条斜率为 $-\alpha$ 的直线。线性关系 $y=ax+b$ 的特点是 $x$ 每增加同样的量，$y$ 改变同样的量；幂律的特点则是 $x$ 每乘以同样的倍数，$y$ 乘以同样的比例：

$$
\frac{y(cx)}{y(x)}=c^{-\alpha}.
$$

例如 $\alpha=\tfrac12$ 时，将资源增加到 4 倍，对应的幂律项减半。也可写成 $\frac{d\log y}{d\log x}=-\alpha$，把指数理解为相对变化之间的比例。

损失往往包含不可消除的底项，例如

$$
L(N)=L_\infty+aN^{-\alpha},\qquad
L(N,D)=L_\infty+aN^{-\alpha}+bD^{-\beta}.
$$

这里 $N$ 可表示模型参数量，$D$ 表示训练 token 数。若 $L_\infty\ne0$，直接画 $\log L$ 对 $\log N$ 未必是直线；单项模型应考察 $\log(L-L_\infty)$。多变量形式还需要控制另一个变量，或共同拟合，不能把几组不同数据条件下的点直接当作单变量幂律。

<strong class="critical-term">幂律关系不等于幂律概率分布。</strong>前者是两个量之间的函数关系；后者还必须指定随机变量、支持集及归一化常数。例如 $p(x)=Cx^{-\gamma}$ 在 $x\geq x_{\min}>0$ 上成为密度，需要 $\gamma>1$ 且 $C=(\gamma-1)x_{\min}^{\gamma-1}$。Scaling Law 通常讨论损失的经验函数关系，不是在说损失本身服从这种密度。有限范围内拟合良好也不保证任意外推都成立。

### 凸分析与约束优化

<span id="app-math-optimization"></span>

#### 问题形式：变量、目标、约束与可行域

为了统一符号，先把问题写成最小化形式：

<div class="key-formula">

$$
\begin{aligned}
\operatorname*{minimize}_{x\in\mathcal D}\quad&f_0(x)\\
\text{subject to}\quad&g_i(x)\leq0,\quad i=1,\ldots,m,\\
&h_j(x)=0,\quad j=1,\ldots,r.
\end{aligned}
$$

</div>

$x$ 是待求变量，$f_0$ 是目标函数，$\mathcal D$ 是这些函数的共同定义域。满足全部约束的点组成可行域，最优值记为 $p_*$；用下确界定义 $p_*$ 时，不预设最优点一定存在。最大化奖励 $R(x)$ 可以等价改成最小化 $-R(x)$，但转换后梯度和拉格朗日函数的符号必须一起调整。

无约束梯度下降的更新为 $x_{t+1}=x_t-\eta_t\nabla f_0(x_t)$。一阶近似给出

$$
f_0(x-\eta\nabla f_0(x))
=f_0(x)-\eta\|\nabla f_0(x)\|^2+o(\eta),
$$

说明非零梯度处的负梯度是局部下降方向；步长过大仍可能使真实函数上升。有约束时，沿这个方向走还可能离开可行域，因此不能仅求 $\nabla f_0=0$。以下微分形式默认变量位于开定义域内；额外的边界限制统一写入约束，避免遗漏边界的最优性条件。

#### 什么使一个优化问题成为凸问题

<span id="app-math-convexity"></span> 集合 $C$ 是凸集，是指任取 $x,y\in C$ 与 $t\in[0,1]$，线段上的点 $(1-t)x+ty$ 仍在 $C$ 内。定义在凸集上的函数 $f$ 是凸函数，是指

$$
f((1-t)x+ty)\leq(1-t)f(x)+tf(y).
$$

图像位于两点连线下方。严格凸要求对不同点和 $0<t<1$ 严格小于；它保证最优点若存在则唯一，但不保证最优点存在。

对可微凸函数，一个等价判据是

$$
f(y)\geq f(x)+\nabla f(x)^\top(y-x),
$$

即切平面是全局下界。若 $f$ 在开凸域上二阶连续可导，则凸性等价于<strong class="key-term">整个定义域内</strong> $H_f(x)\succeq0$。仅在一个点 Hessian 半正定，不能推出整个函数凸。

标准凸优化问题要求 $f_0$ 和所有不等式函数 $g_i$ 凸，等式约束 $h_j$ 为仿射函数（形如 $a_j^\top x-b_j$），并具有凸定义域。一般非线性等式即使看似简单，也可能产生非凸可行域，例如 $x^2=1$ 只允许两个分离的点。

在可微凸问题中，可行点 $x_*$ 最优的一个等价条件是

$$
\nabla f_0(x_*)^\top(x-x_*)\geq0\qquad\text{对所有可行 }x.
$$

这表示没有可行方向能在一阶上降低目标。无约束时可向所有方向移动，该条件退化为梯度为零；有约束时梯度可以由边界“挡住”。凸问题的局部最优也是全局最优。神经网络训练通常非凸，但其中的局部二次子问题、分布优化或线性约束问题可能是凸的，不能把子问题的性质直接推广到全部参数训练。

#### 等式约束与拉格朗日乘子

<span id="app-math-lagrange"></span> 先考虑一个等式 $h(x)=0$。在约束梯度非零的规则点处，可行曲面的切向扰动 $v$ 满足 $\nabla h(x)^\top v=0$。若 $x_*$ 最优，目标沿所有切向方向的一阶变化都应为零，所以 $\nabla f_0(x_*)$ 必须位于约束的法向方向上。

多个等式的情形，在约束梯度满足适当独立性条件时，可写成

$$
\nabla f_0(x_*)+\sum_j\nu_j\nabla h_j(x_*)=0.
$$

引入<strong class="key-term">拉格朗日函数</strong>

$$
\mathscr L(x,\nu)=f_0(x)+\sum_j\nu_jh_j(x),
$$

便得到方程组 $\nabla_x\mathscr L=0$、$h_j(x)=0$。等式乘子 $\nu_j$ 可以为任意实数。它们把受约束的驻点问题转化成联立求解，但对非凸问题，解出方程组后仍需判断极值类型。

###### 例：直线上离原点最近的点

考虑 $\min_{x,y}\frac12(x^2+y^2)$，约束 $x+y=1$。构造

$$
\mathscr L(x,y,\nu)=\tfrac12(x^2+y^2)+\nu(x+y-1).
$$

驻点与可行性方程为

$$
x+\nu=0,\qquad y+\nu=0,\qquad x+y=1,
$$

因此 $x_*=y_*=\tfrac12$，$\nu_*=-\tfrac12$，最优值 $p_*=\tfrac14$。目标严格凸、约束仿射，所以这是唯一全局最优点。也可消元 $y=1-x$ 后直接求导，得到相同答案。

#### 不等式乘子与对偶函数

<span id="app-math-duality"></span> 恢复不等式约束后，定义

<div class="key-formula">

$$
\mathscr L(x,\lambda,\nu)
=f_0(x)+\sum_i\lambda_i g_i(x)+\sum_j\nu_jh_j(x),
\qquad\lambda_i\geq0.
$$

</div>

为什么不等式乘子必须非负？对最小化问题的任意可行点，$g_i(x)\leq0$、$h_j(x)=0$，因此

$$
\mathscr L(x,\lambda,\nu)\leq f_0(x).
$$

这个下界关系决定了符号约定。如果把约束改写成 $g_i(x)\geq0$，拉格朗日项的符号也要改变。

固定乘子，暂时放开显式约束，只保留函数定义域，得到<strong class="key-term">对偶函数</strong>

$$
q(\lambda,\nu)=\inf_{x\in\mathcal D}\mathscr L(x,\lambda,\nu).
$$

它有时为 $-\infty$。无论原问题是否凸，$q$ 都是乘子的凹函数，因为它是关于乘子的仿射函数族的逐点下确界。

对偶问题寻找最紧的这种下界：

$$
d_*=\sup_{\lambda\geq0,\nu}q(\lambda,\nu).
$$

原问题在可行变量中找最小目标值，对偶问题在乘子中找最大下界。不要把 $\inf_x\sup_{\lambda,\nu}$ 与 $\sup_{\lambda,\nu}\inf_x$ 随意交换；它们是否给出相同值，正是强对偶所讨论的问题。

#### 弱对偶、强对偶与 Slater 条件

###### 弱对偶为什么总能提供下界

对任意原始可行点 $\tilde x$ 和任意 $\lambda\geq0$、$\nu$，

$$
q(\lambda,\nu)
\leq\mathscr L(\tilde x,\lambda,\nu)
\leq f_0(\tilde x).
$$

分别优化两边，就得到<strong class="key-term">弱对偶</strong> $d_*\leq p_*$。这不要求原问题凸。当找到一对原始、对偶可行解时，有限的差值 $f_0(\tilde x)-q(\lambda,\nu)$ 给出当前目标值距离最优值的上界；差值为零就能证明全局最优。

###### 强对偶需要额外条件

若 $d_*=p_*$，称为<strong class="key-term">强对偶</strong>；若两者有限，$p_*-d_*$ 称为对偶间隙。凸性本身并不在所有退化情形下自动保证强对偶和乘子解存在。

一个常用的充分条件是 <strong class="key-term">Slater 条件</strong>。在本节采用的光滑凸问题、开凸共同定义域内，若存在一点 $\bar x$ 满足

$$
g_i(\bar x)<0\quad\text{对所有不等式},\qquad
h_j(\bar x)=0\quad\text{对所有等式},
$$

即存在严格可行点，则在最优值有限等通常条件下，强对偶成立，并存在最优对偶乘子。更一般的表述使用共同定义域的相对内部。Slater 是充分条件，未满足它不等于强对偶一定失败。

乘子还可解释为约束的边际代价。例如把约束写成 $g_i(x)\leq u_i$，在最优值函数可微且相关正则条件成立时，放宽该约束的边际影响为 $\partial p_*/\partial u_i=-\lambda_i^*$。较大的乘子意味着轻微放宽这一约束，可能带来更大的目标改善；没有可微性时，不能不加条件地把它当成普通导数。

#### KKT 条件：可行性、平衡与互补松弛

<span id="app-math-kkt"></span> 对可微问题，Karush–Kuhn–Tucker（KKT）条件由四部分组成：

<div class="key-formula">

$$
\begin{aligned}
\text{原始可行性：}\quad&g_i(x_*)\leq0,\quad h_j(x_*)=0;\\
\text{对偶可行性：}\quad&\lambda_i^*\geq0;\\
\text{互补松弛：}\quad&\lambda_i^*g_i(x_*)=0;\\
\text{驻点条件：}\quad&\nabla f_0(x_*)+\sum_i\lambda_i^*\nabla g_i(x_*)
+\sum_j\nu_j^*\nabla h_j(x_*)=0.
\end{aligned}
$$

</div>

原始可行性保证变量没有违规，对偶可行性保证乘子符号正确，驻点条件说明目标梯度与约束梯度相互平衡。互补松弛则区分约束是否真正起作用：

- 若 $g_i(x_*)<0$，约束未激活，必有 $\lambda_i^*=0$。

- 若 $\lambda_i^*>0$，必有 $g_i(x_*)=0$，约束处于边界。

- 反过来，$g_i(x_*)=0$ 并不保证乘子严格为正；激活的约束也可能对应零乘子。

在强对偶且原始、对偶最优解均达到时，

$$
q(\lambda^*,\nu^*)\leq\mathscr L(x_*,\lambda^*,\nu^*)\leq f_0(x_*)
$$

两端相等，因此中间也取等号。第二个不等式取等号给出互补松弛；第一个取等号意味着 $x_*$ 最小化对应的拉格朗日函数，在可微内点处给出驻点条件。

###### 何时必要，何时充分

对一般非凸问题，在局部最优点满足约束资格条件时，KKT 是必要条件。例如一种常见资格条件是：所有等式约束梯度和激活不等式约束梯度线性无关。没有这些前提，局部最优点可能根本找不到 KKT 乘子。

对凸问题，任意满足 KKT 的点都是全局最优点：此时 $\mathscr L$ 关于 $x$ 凸，驻点即其全局最小值，再用互补松弛即可让上下界相等。若进一步满足 Slater 条件并且原始最优点存在，KKT 就可作为最优性的充要刻画。非凸问题即使满足 KKT，也可能只是局部极值或鞍点。

###### 退化例子：最优点存在，但 KKT 乘子不存在

考虑 $\min_x x$，约束 $x^2\leq0$。唯一可行点是 $x_*=0$，显然最优；但驻点条件要求 $1+2\lambda x_*=0$，任何有限 $\lambda$ 都无法满足。这里约束梯度在可行点为零，也不存在严格可行点。

这个例子甚至仍有强对偶：$\lambda>0$ 时

$$
q(\lambda)=\inf_x(x+\lambda x^2)=-\frac1{4\lambda},
$$

其上确界为 0，与原始最优值相等，但只能在 $\lambda\to\infty$ 时逼近，没有有限的最优乘子。这说明强对偶、对偶最优解达到、KKT 乘子存在是需要分别检查的陈述。

#### 完整例题：带上界的平方损失

<span id="app-math-kkt-example"></span> 考虑一个可直接画图理解的问题：

$$
\min_x\frac12(x-a)^2\qquad\text{subject to }x\leq b.
$$

无约束最小值在 $a$；约束把允许区域截在 $b$ 左侧。令 $g(x)=x-b$，构造

$$
\mathscr L(x,\lambda)=\frac12(x-a)^2+\lambda(x-b).
$$

KKT 条件为

$$
x\leq b,\quad\lambda\geq0,\quad\lambda(x-b)=0,\quad x-a+\lambda=0.
$$

<strong class="list-label">情况一：</strong>$a\leq b$。取 $x_*=a$、$\lambda_*=0$ 即可满足全部条件；当 $a=b$ 时约束激活但乘子为零。

<strong class="list-label">情况二：</strong>$a>b$。若乘子为零则会得到不可行的 $x=a$，因此必须在边界 $x_*=b$，驻点条件给出 $\lambda_*=a-b>0$。综上，

$$
\boxed{x_*=\min(a,b),\qquad\lambda_*=(a-b)_+,\qquad
p_*=\tfrac12(a-b)_+^2,}
$$

其中 $(t)_+=\max(t,0)$。

也可独立求对偶。固定 $\lambda$ 时，最小化 $\mathscr L$ 得到 $x=a-\lambda$，代回可得

$$
q(\lambda)=\lambda(a-b)-\tfrac12\lambda^2,
\qquad\max_{\lambda\geq0}q(\lambda).
$$

它是一维凹二次函数，最优解正是 $\lambda_*=(a-b)_+$，且 $d_*=p_*$。例如 $a=2,b=1$ 时，$x_*=1$、$\lambda_*=1$、原始与对偶目标均为 $1/2$。该问题有严格可行点 $x<b$，所以也符合 Slater 条件。这一例子同时展示了约束激活、乘子、对偶下界和强对偶，而不是只列出四条方程。

#### 实际怎样求解约束问题

<span id="app-math-constrained-methods"></span> KKT 是描述最优解的条件，并非一套对任意规模问题都能直接运行的算法。小问题可枚举可能激活的约束，再联立驻点、可行性与互补松弛方程；大问题通常利用以下结构选择数值方法。

###### 消元与投影

简单等式可以先消元。若 $Ax=b$ 有特解 $x_p$，且 $Z$ 的列张成 $A$ 的零空间，可令 $x=x_p+Zz$，把问题改写成对 $z$ 的无等式约束优化。

若可行域 $C$ 的投影容易计算，可以使用投影梯度法：

$$
x_{t+1}=\Pi_C(x_t-\eta_t\nabla f_0(x_t)),\qquad
\Pi_C(z)=\operatorname*{arg\,min}_{x\in C}\tfrac12\|x-z\|_2^2.
$$

对非空闭凸集投影唯一；区间约束可通过截断实现。一般约束集的投影本身可能就是一个困难的优化问题。收敛还需要结合目标光滑性、步长等条件判断。

###### 罚函数与增广拉格朗日

一种直接方法是在目标中惩罚违反约束的程度，例如

$$
f_0(x)+\frac\rho2\sum_jh_j(x)^2
+\frac\rho2\sum_i\max(0,g_i(x))^2.
$$

有限的 $\rho$ 通常只让违规变小，并不保证约束严格满足；不断增大 $\rho$ 还可能使数值条件变差。因而不能把任意固定的惩罚系数都解释为精确的约束乘子。

对等式约束，增广拉格朗日同时保留线性乘子项与二次惩罚：

$$
\mathscr L_\rho(x,\nu)=f_0(x)+\nu^\top h(x)+\frac\rho2\|h(x)\|_2^2.
$$

常见迭代是先近似最小化 $\mathscr L_\rho(x,\nu_t)$，再更新 $\nu_{t+1}=\nu_t+\rho h(x_{t+1})$。乘子根据约束残差调整；这与固定一个很大的二次罚系数有所区别。具体收敛性质取决于问题和内部求解精度。

###### 障碍法与内点思想

从严格可行点出发，可用对数障碍防止越过不等式边界：

$$
\min_x\ f_0(x)-\mu\sum_i\log(-g_i(x)),\qquad h_j(x)=0,
$$

其中 $g_i(x)<0$、$\mu>0$。接近边界时障碍项趋于正无穷，逐步减小 $\mu$ 则允许解靠近真实最优边界。障碍驻点中出现 $\lambda_i=\mu/(-g_i(x))>0$，满足

$$
\lambda_i g_i(x)=-\mu.
$$

它是互补松弛的平滑近似；在合适条件下，$\mu\to0$ 时逐渐接近 KKT 条件。内点法利用这一结构求解连续的一系列子问题，而不是直接跳到边界上。

###### 原始与对偶变量交替更新

在可微情形，一种基本思路是对 $x$ 下降、对乘子上升：

$$
\begin{aligned}
x_{t+1}&=x_t-\eta_t\nabla_x\mathscr L(x_t,\lambda_t,\nu_t),\\
\lambda_{t+1}&=[\lambda_t+\alpha_t g(x_{t+1})]_+,\\
\nu_{t+1}&=\nu_t+\alpha_t h(x_{t+1}).
\end{aligned}
$$

其中非负截断逐元素作用于 $\lambda$；必要时还需把 $x$ 投影回定义域。违反 $g_i\leq0$ 时，乘子会上升，加大该约束的影响。这些式子说明更新方向，但不是对所有问题都收敛的承诺：鞍点迭代可能振荡，需要适当步长、额外梯度或更稳定的子问题求解方式。

#### 正文连接一：Fisher 信息与 TRPO 的局部约束

<span id="app-math-fisher-trpo"></span>

###### Fisher 为什么是半正定矩阵

给定分布 $p_\theta(z)$，定义得分向量（score）$s_\theta(z)=\nabla_\theta\log p_\theta(z)$。在支持集不随参数变化、允许交换积分与微分等正则条件下，

$$
\mathbb E_{p_\theta}[s_\theta(Z)]
=\int\nabla_\theta p_\theta(z)\,dz
=\nabla_\theta1=0.
$$

Fisher 信息矩阵定义为

$$
F(\theta)=\mathbb E_{p_\theta}[s_\theta(Z)s_\theta(Z)^\top].
$$

对任意 $v$，$v^\top Fv=\mathbb E[(v^\top s_\theta)^2]\geq0$，故它总是半正定，但可能奇异。进一步满足二阶微分所需条件时，还可得到

$$
F(\theta)=-\mathbb E_{p_\theta}[\nabla_\theta^2\log p_\theta(Z)].
$$

这个等式是模型分布下的期望恒等式；不能直接把任意有限数据上的损失 Hessian、梯度外积与 Fisher 都视为同一个矩阵。

###### KL 的局部二次近似

以固定旧参数 $\theta_0$ 为参照，定义

$$
K(\theta)=D_{\mathrm{KL}}(p_{\theta_0}\|p_\theta)
=\mathbb E_{p_{\theta_0}}[\log p_{\theta_0}(Z)-\log p_\theta(Z)].
$$

在 $\theta=\theta_0$ 处，$K=0$、$\nabla K=0$，Hessian 为 $F(\theta_0)$。所以对小更新 $\Delta$，

$$
K(\theta_0+\Delta)=\tfrac12\Delta^\top F(\theta_0)\Delta
+o(\|\Delta\|^2).
$$

TRPO 中对固定的旧策略状态分布取平均，并对各状态的动作分布计算 KL，可得到相应的平均 Fisher。它衡量参数移动引起的分布变化，而不只是参数欧氏距离的大小。

###### 用乘子推导信赖域方向

把待最大化的局部代理目标一阶展开，令其梯度为 $g$，再把 KL 约束二阶展开，就得到子问题

$$
\max_\Delta g^\top\Delta
\qquad\text{subject to }\tfrac12\Delta^\top F\Delta\leq\delta.
$$

先假设 $F\succ0$、$g\ne0$、$\delta>0$。将其改成最小化 $-g^\top\Delta$，拉格朗日函数为

$$
\mathscr L(\Delta,\lambda)=-g^\top\Delta
+\lambda\bigl(\tfrac12\Delta^\top F\Delta-\delta\bigr).
$$

驻点条件为 $-g+\lambda F\Delta=0$，因此 $\Delta=\lambda^{-1}F^{-1}g$。目标是非零线性函数，最优点必在约束边界；代入边界等式得到

$$
\lambda=\sqrt{\frac{g^\top F^{-1}g}{2\delta}},\qquad
\boxed{\Delta_*=
\sqrt{\frac{2\delta}{g^\top F^{-1}g}}\,F^{-1}g.}
$$

$F^{-1}g$ 是自然梯度方向，前面的因子按 KL 预算缩放步长。若 $g=0$，一阶模型无法区分方向，可选择零步长。若 $F$ 仅半正定，逆矩阵未必存在；若 $g$ 在 $F$ 的零空间方向上有非零分量，这个局部子问题甚至可能无界。

实现中常用 $F+\varepsilon I$ 做阻尼，并通过共轭梯度等方法近似求解线性方程，用矩阵向量积避免存储完整矩阵。阻尼会改变局部度量；求得的步长还需通过线搜索检查实际代理目标与 KL。上述闭式式子求解的是<strong class="key-term">局部近似子问题</strong>，并不保证原始神经网络约束问题的全局最优。

#### 正文连接二：KL 正则下的最优策略与 DPO

<span id="app-math-kl-policy"></span> 固定一个提示，考虑有限候选回答 $i=1,\ldots,K$。参考分布 $q_i>0$、$\sum_iq_i=1$，奖励 $r_i$ 有限，$\beta>0$。在所有概率向量 $p$ 上求解

$$
\max_p\left\{\sum_i p_ir_i-\beta\sum_ip_i\log\frac{p_i}{q_i}\right\},
\quad p_i\geq0,\quad\sum_i p_i=1.
$$

取约定 $0\log0=0$。这是单纯形上的凹最大化问题；在内部，目标 Hessian 为 $-\beta\operatorname{diag}(1/p_i)$，严格负定。下面先求一个内部候选解，再证明它也是包括边界在内的唯一最优解。

对归一化约束引入乘子 $\nu$，采用最大化形式

$$
\mathscr J(p,\nu)=\sum_i p_ir_i-\beta\sum_i p_i\log\frac{p_i}{q_i}
+\nu\left(\sum_ip_i-1\right).
$$

令内部一阶导数为零：

$$
r_i-\beta\left(\log\frac{p_i}{q_i}+1\right)+\nu=0.
$$

整理并利用 $\sum_ip_i=1$，得到

<div class="key-formula">

$$
p_i^*=\frac{q_i e^{r_i/\beta}}{Z},\qquad
Z=\sum_jq_j e^{r_j/\beta}.
$$

</div>

该解所有分量均为正，满足可行性。把 $\log p_i^*=\log q_i+r_i/\beta-\log Z$ 代回原目标，对任意可行 $p$ 有

$$
\sum_i p_ir_i-\beta D_{\mathrm{KL}}(p\|q)
=\beta\log Z-\beta D_{\mathrm{KL}}(p\|p^*)
\leq\beta\log Z.
$$

等号仅在 $p=p^*$ 时成立，因而得到全局最优性与唯一性。参考分布为零的候选不在这里的假设内；有限 KL 的解不能任意把质量放到参考分布没有支持的位置。

将最优解反解为奖励，

$$
r_i=\beta\log\frac{p_i^*}{q_i}+\beta\log Z.
$$

对同一个提示下的两个回答作差，归一化项 $\beta\log Z$ 消失。这正是正文 DPO 推导中用策略与参考策略的对数概率比表示奖励差的关键代数关系；再结合偏好概率模型，才能形成实际的偏好学习目标。

这里优化的是自由概率分布 $p$。将它限制为共享神经网络参数所表达的 $p_\theta$ 后，训练问题通常不再凸，不能沿用上述全局最优保证。固定的 KL 惩罚系数 $\beta$ 也不等于任意给定的 KL 上界 $\delta$：只有在对应约束问题满足适当对偶条件，并选择合适的最优乘子时，两种形式才有相应联系。

### 参考文献与延伸阅读

<strong class="note-label">正文回查：</strong>[损失、优化器与矩阵更新](/part-1/chapter-3#guide-ch-3)；[采样与解码](/part-1/chapter-5#guide-ch-5)；[幂律拟合](/part-3/chapter-9#guide-ch-10)；[TRPO 补充推导](/part-5/chapter-13#guide-ch-12-grpo)；[DPO 的奖励与策略关系](/part-6/chapter-14#guide-ch-13)。

<strong class="note-label">延伸阅读：</strong>[Petersen 与 Pedersen：The Matrix Cookbook](https://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf)（矩阵微分公式）； [Boyd 与 Vandenberghe：Convex Optimization，Duality 讲义](https://web.stanford.edu/class/ee364a/lectures/duality.pdf)（对偶与 KKT）； [Schulman 等：Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)（信赖域策略优化）。本附录统一使用列梯度与 Frobenius 梯度约定，阅读其他资料时先核对符号和定义域。

## 深度学习基础

<span id="app-deep-learning"></span> 本附录用常见任务说明深度学习在做什么，再介绍搭建网络和完成训练所需的基本组件。阅读正文时，可以先在这里建立整体印象，再回到相应章节学习具体模型与实现；矩阵求导、最大似然等数学推导见[数学基础](/appendix#app-math)。

### 深度学习在学什么

<span id="app-deep-learning-concepts"></span> <strong class="key-term">机器学习</strong>通过数据调整模型，使它能对新样本作出预测。<strong class="key-term">深度学习</strong>是其中以多层神经网络学习表示和映射关系的一类方法：输入经过一层层变换，逐渐形成适合当前任务的内部表示，最后产生预测结果。这里的“深”主要指变换层次较多，而不是对模型是否具有理解能力的判断。

把网络记为 $\hat y=f_\theta(x)$，$x$ 是输入，$\hat y$ 是预测，$\theta$ 包含网络中需要学习的权重和偏置。训练时，用<strong class="key-term">损失函数</strong>衡量预测与目标的差距，再根据梯度调整参数。一个最基本的训练过程是

$$
\text{输入数据}\ \longrightarrow\ \text{网络预测}\ \longrightarrow\
\text{计算损失}\ \longrightarrow\ \text{反向传播}\ \longrightarrow\ \text{更新参数}.
$$

例如识别手写数字时，输入是图像像素，输出是 0 到 9 各类别的分数。训练会调整网络，使正确类别获得更高的预测概率。中间层可以学习局部笔画、形状组合等有用表示，但这些表示由训练任务和数据共同决定，不必与人类命名的概念逐一对应。

###### 参数、超参数与训练结果

权重、偏置以及嵌入表中的数值通常由训练学习，称为<strong class="key-term">参数</strong>；学习率、层数、隐藏维度、批量大小等由训练方案指定，称为<strong class="key-term">超参数</strong>。损失下降说明模型更好地拟合了当前训练目标，是否能处理未见过的数据，还要通过独立的验证集和测试集判断。

###### 监督学习、自监督学习与强化学习

监督学习使用给定的输入与目标，例如“图片—类别”或“语音—转写文本”。自监督学习从数据本身构造预测目标，例如用文本前缀预测下一个 token，因此不必逐条人工标注类别，但仍有明确的训练目标。强化学习则根据交互产生的奖励改进决策。它们描述的是学习信号的来源，神经网络可以作为这些方法共同使用的模型。

### 几个典型案例

<span id="app-deep-learning-examples"></span>

- <strong class="list-label">图像分类：</strong>输入一张手写数字或物体图片，输出类别。卷积网络利用局部区域与共享参数提取视觉特征，再通过分类层给出结果。这个案例适合串起“输入、隐藏层、损失、训练与测试”的完整流程。

- <strong class="list-label">数值预测：</strong>输入房屋面积、位置等特征，预测价格，或者根据历史观测预测未来需求。输出是连续数值，常见起点是线性模型或多层感知机，损失可以使用均方误差。是否需要更复杂的网络，要看数据规模与任务表现。

- <strong class="list-label">语音识别与机器翻译：</strong>输入和输出都是有顺序的序列，而且长度可能不同。循环网络或注意力模型处理上下文关系，编码器提取输入表示，解码器据此产生输出序列。

- <strong class="list-label">语言模型与文本生成：</strong>输入文本前缀，预测词表中下一个 token 的分布。训练时目标 token 来自已有文本；生成时则选择或采样一个 token，接到前缀后继续预测。正文讨论的 Transformer 语言模型属于这一类。

这些案例可从三个问题入手比较：输入如何表示，输出是什么，怎样定义“预测得好”。网络组件与损失的选择，都应服务于这三个问题。

### 深度学习中常用的组件

<span id="app-deep-learning-components"></span> 一个网络通常由若干组件组合而成。下面介绍各组件解决的问题，不要求每个模型都包含它们。

#### 线性层、激活函数与多层感知机

<strong class="key-term">线性层（全连接层）</strong>对输入特征作加权组合。采用列向量记法，输入 $x\in\mathbb R^d$、权重 $W\in\mathbb R^{h\times d}$、偏置 $b\in\mathbb R^h$，输出为

$$
z=Wx+b,\qquad a=\phi(z).
$$

其中 $\phi$ 是逐元素的<strong class="key-term">激活函数</strong>，负责引入非线性。仅把多个仿射变换串起来，结果仍可合并为一个仿射变换；加入非线性后，多层网络才能表达更丰富的关系。

ReLU 的定义是 $\max(0,z)$；Sigmoid 把实数映射到 $(0,1)$，常用于二分类输出或门控；GELU、SiLU 也是常见激活。SiLU 为 $z\,\operatorname{sigmoid}(z)$。这些函数的形状与梯度不同，不能只看名称就认为可以无影响地互换。

<strong class="key-term">多层感知机（MLP）</strong>把线性层与激活函数交替堆叠，例如

$$
f(x)=W_2\phi(W_1x+b_1)+b_2.
$$

它既可独立完成分类、回归，也可作为大网络中的子模块。正文 Transformer 的前馈网络在每个 token 位置上变换特征；SwiGLU 则进一步用一条分支生成门控信号，与另一条分支逐元素相乘。

#### 嵌入层：把离散符号变成向量

token ID、类别 ID 等整数首先表示符号身份，不宜直接把编号大小当作语义远近。<strong class="key-term">嵌入层（Embedding）</strong>建立一个可学习的向量表 $E\in\mathbb R^{V\times d}$，用 ID 取出对应行，作为该符号的 $d$ 维表示。$V$ 是符号表大小，$d$ 是表示维度。

在语言模型中，分词器先把文本变成 token ID，嵌入层再把它们变成向量。嵌入随训练更新；若只使用 token 嵌入，网络还没有直接获得各 token 的顺序，因此正文还会介绍位置编码与 RoPE。分词、嵌入和位置处理承担不同职责。

#### 卷积、池化与循环结构

<strong class="key-term">卷积层</strong>用一组可学习的卷积核处理局部邻域，并在不同位置共享权重，适合利用图像等数据的局部结构。<strong class="key-term">池化层</strong>对邻域取最大值或平均值，常用于缩小空间尺寸、聚合局部信息；它也会丢失部分细节，并非所有网络都必须使用。

<strong class="key-term">循环神经网络（RNN）</strong>把前一步的隐藏状态传到下一步，使当前表示依赖历史输入。LSTM 和 GRU 通过门控控制信息保留与更新，用于缓解普通循环网络处理长依赖时的困难。了解它们有助于理解序列建模的思路；正文的 Transformer 则主要依靠注意力建立位置之间的联系。

#### 注意力：按内容聚合上下文

<strong class="key-term">注意力（Attention）</strong>让一个位置根据内容，从其他位置选择并汇总信息。通常把当前需求表示为查询 $Q$，把可供匹配的信息表示为键 $K$，把实际要汇总的内容表示为值 $V$；匹配分数经过归一化后，用来对值向量加权求和。

自注意力中的查询、键和值来自同一序列，多头注意力则允许多个子空间分别建立联系。因果掩码限制一个位置只能使用允许的前缀，避免预测下一个 token 时看到未来答案。注意力负责位置之间的信息交流，前馈网络负责逐位置的特征变换，两者在 Transformer 中配合使用。

#### 残差连接与归一化

<strong class="key-term">残差连接</strong>把子模块输出与输入相加，即 $y=x+F(x)$，要求两者形状匹配，必要时可先作投影。它为信息与梯度提供直接通路，使深层网络更容易优化；并不意味着任意增加深度都能自动获得更好结果。

<strong class="key-term">归一化层</strong>根据选定维度上的统计量调整激活尺度，通常再配合可学习的缩放或偏移。BatchNorm 在训练时使用批次等维度上的统计量，推理时通常使用累计统计量；LayerNorm 通常在单个样本或 token 的特征维度上计算均值与方差；RMSNorm 根据均方根缩放，省去减均值这一步。正文会进一步讨论 RMSNorm，以及归一化放在子模块之前或之后的区别。

#### 输出层、损失函数与优化器

分类网络常先输出未经归一化的分数，称为 <strong class="key-term">logits</strong>。Softmax 将一组 logits 转成总和为 1 的类别概率；语言模型的输出层就要为词表中的每个 token 产生一个分数。回归任务则可以直接输出所预测的数值，不必使用 Softmax。

<strong class="key-term">损失函数</strong>把预测好坏变成可优化的标量。均方误差常用于数值回归，交叉熵常用于分类与下一个 token 预测。损失与评测指标未必相同，例如训练用交叉熵，评测还可能关心准确率或生成质量。Softmax 与交叉熵的数学关系见[数学基础](/appendix#app-math)，正文再展开语言模型的具体目标。

<strong class="key-term">优化器</strong>根据梯度更新参数。SGD 使用当前批次的梯度，带动量的方法积累更新趋势，Adam 类方法还维护梯度的一阶、二阶矩估计。学习率决定每次更新的尺度，调度器可让它随训练进度变化。优化器、学习率与初始化需要配合；它们不会改变损失函数本身的定义。

### 阅读正文时需要补足的几个概念

<span id="app-deep-learning-training"></span>

#### 张量形状与自动求导

<span id="app-tensors"></span> 深度学习框架用<strong class="key-term">张量</strong>存放数据、参数和中间结果。设批量大小为 $B$、序列长度为 $T$、隐藏维度为 $d$、词表大小为 $V$，语言模型的一条典型数据路径是

$$
\underbrace{(B,T)}_{\text{token ID}}
\ \longrightarrow\ \underbrace{(B,T,d)}_{\text{隐藏表示}}
\ \longrightarrow\ \underbrace{(B,T,V)}_{\text{logits}}.
$$

读公式和实现时，先认清每一维的含义。广播会让长度为 1 或缺失的维度扩展参与运算，例如 $(B,T,d)+(d)$ 可以给每个 token 加同一个偏置；Softmax、求和与平均则必须明确沿哪一维进行。转置交换维度顺序，重塑改变形状但保留元素数量，二者也不等价。

<strong class="key-term">前向传播</strong>计算预测和损失；<strong class="key-term">反向传播</strong>按链式法则计算梯度；<strong class="key-term">自动求导</strong>由框架记录计算图并完成这些导数计算。反向传播得到梯度后，还需要优化器执行参数更新。在 PyTorch 中梯度会累加到相应参数的 `.grad`，因此通常要在下一轮累计之前清零。

#### 批量、训练步与梯度累积

一个 mini-batch 是一次参与计算的一组样本；遍历训练集一轮通常称为一个 epoch。阅读训练记录时要确认“step”指一次前向与反向计算，还是一次优化器更新，因为采用梯度累积后两者数量不同。

<strong class="key-term">梯度累积</strong>把多个小批次的梯度合并后再更新参数，可在显存受限时实现更大的有效批量。累积时应统一损失的求和、平均及缩放方式；遇到不同序列长度，还要明确是按序列还是按有效 token 平均。它与计算图是否保留是不同的问题。

#### 泛化、正则化与训练模式

<strong class="key-term">过拟合</strong>表现为训练数据上的效果不断改善，而未见数据上的效果没有同步改善，甚至变差。训练集用于学习参数，验证集用于选择方案，测试集用于最终评估；把测试结果反复用于调参，会使其失去独立评估的意义。

数据增强、权重衰减、Dropout 和早停是常见的控制过拟合手段。<strong class="key-term">Dropout</strong>在训练时随机屏蔽部分激活，并按实现约定调整尺度；评估时关闭这种随机屏蔽。<strong class="key-term">权重衰减</strong>在更新时抑制权重幅度；它与在损失中加入 $L_2$ 惩罚的对应关系依赖优化器，不能对 Adam 等方法直接混同。它们是否适合某个模型，应结合验证表现判断。

在 PyTorch 中，`model.eval()` 改变 Dropout、BatchNorm 等模块的训练或评估行为，但不会自动关闭梯度；`no_grad` 关闭相关计算的梯度记录；参数的 `requires_grad` 决定是否需要对该参数求导。正文冻结参考模型、执行生成或评测时，会用到这些区别。

#### 从预训练到微调

<strong class="key-term">预训练</strong>通常在大规模数据上学习通用表示或预测能力，<strong class="key-term">微调</strong>再从已有参数出发继续训练。正文中，语言模型预训练预测下一个 token，指令微调学习目标回答，偏好优化利用回答之间的偏好信号；它们仍沿用前向计算、计算损失、反向传播与参数更新的训练流程。

<strong class="note-label">接回正文：</strong>组件实现见[Transformer 架构](/part-1/chapter-2#guide-ch-2)；损失与优化器见[语言模型的训练](/part-1/chapter-3#guide-ch-3)；生成过程见[推理与采样](/part-1/chapter-5#guide-ch-5)；微调目标见[指令微调与偏好对齐理论](/part-6/chapter-14#guide-ch-13)。

### 参考文献与延伸阅读

<strong class="list-label">黄金参考教材：</strong>[邱锡鹏《神经网络与深度学习》](https://github.com/nndl/nndl)。AI 学习的启蒙教材，适合梳理概念与数学推导。

<strong class="list-label">白银参考教材：</strong>[李沐等《动手学深度学习》](https://zh.d2l.ai/)。适合结合代码学习模型与训练。

## AI Infra 基础

<span id="app-infra"></span> 本附录汇集混合精度训练、GPU 内存架构与 CUDA 通信机制，供阅读[性能分析与基准测试](/part-2/chapter-6#guide-ch-7)、[FlashAttention 与 Triton 优化](/part-2/chapter-7#guide-ch-8)和[分布式训练与并行策略](/part-2/chapter-8#guide-ch-9)时查阅。

### GPU 系统知识

<span id="app-gpu"></span>

#### GPU 内存架构

<span id="sec-8-1"></span><span id="app-gpu-memory"></span>

[参考资料 ](/appendix#read-8-1)

##### 物理内存架构：片上内存与片下内存

<span id="sec-8-1-1"></span>

从硬件位置来看,GPU内存主要分为两类:<strong class="key-term">片上(on-chip)内存</strong>和<strong class="key-term">片下(off-chip)内存</strong> 。

- <strong class="critical-term">片上内存 (On-chip Memory)</strong>:位于GPU芯片内部,主要用作高速缓存(Cache)、共享内存(Shared Memory)和寄存器(Register)。

  - <strong class="list-label">特点</strong>:速度极快，但存储空间非常小 。

  - <strong class="list-label">硬件类型</strong>:通常是<strong class="key-term">SRAM</strong>(静态随机存取存储器)，其优点是访问速度快，无需刷新即可保存数据。

- <strong class="critical-term">片下内存 (Off-chip Memory)</strong>:位于GPU芯片外部,主要用作全局内存(Global Memory),也就是我们常说的“显存”。

  - <strong class="list-label">特点</strong>:容量大，但速度相对较慢 。

  - <strong class="list-label">硬件类型</strong>:通常是<strong class="key-term">HBM</strong>(高带宽内存),它通过将多个DDR芯片堆叠并与GPU封装在一起,以实现大容量和高位宽。HBM属于<strong class="key-term">DRAM</strong>(动态随机存取存储器),其成本较低、密度高,但访问速度慢于SRAM。

<figure data-latex-placement="H">
<img src="/images/4f69d05823.png" style="width:80.0%" alt="GPU内存架构" />
<figcaption>GPU内存架构</figcaption>
</figure>

##### 逻辑内存层次与功能划分

<span id="sec-8-1-2"></span>

在CUDA编程模型中,GPU内存根据其作用域、生命周期和访问特性被划分为不同的逻辑类型。其速度和容量关系通常是:<strong class="key-term">寄存器 \> 共享内存/L1缓存 \> L2缓存 \> 全局内存</strong>。

以下是各类内存的详细介绍:

1.  <strong class="critical-term">寄存器 (Register)</strong>

    - <strong class="list-label">位置与速度</strong>:位于片上,是GPU中速度最快的内存空间 。

    - <strong class="list-label">作用域</strong>:每个线程独享的私有资源,用于存储线程内频繁使用的临时变量 。

    - <strong class="list-label">生命周期</strong>:与核函数(Kernel)的执行周期一致,核函数运行结束即被释放。

    - <strong class="list-label">特点</strong>:容量非常有限。如果一个线程需要的变量过多,寄存器会发生“溢出”,多出的变量将被存放到速度慢得多的本地内存中,严重影响性能。

2.  <strong class="critical-term">本地内存 (Local Memory)</strong>

    - <strong class="list-label">位置与速度</strong>:物理上它与全局内存位于同一区域,即片下的HBM/DRAM中,因此访问延迟高、速度慢 。

    - <strong class="list-label">作用域</strong>:和寄存器一样，是每个线程的私有内存。

    - <strong class="list-label">用途</strong>:主要用于存放两种数据：① 编译器无法确定索引的本地数组；② 因体积过大或数量过多而无法放入寄存器的变量（即寄存器溢出）。

3.  <strong class="critical-term">共享内存 (Shared Memory)</strong>

    - <strong class="list-label">位置与速度</strong>:位于片上(on-chip),是可编程的内存,访问速度非常快,几乎和寄存器一样 。

    - <strong class="list-label">作用域</strong>:被一个线程块(Block)内的所有线程共享,可用于块内线程间的高效通信 。不同线程块之间无法通过共享内存通信。

    - <strong class="list-label">生命周期</strong>:与线程块的生命周期一致,线程块执行开始时分配,执行结束时释放。

4.  <strong class="critical-term">全局内存 (Global Memory)</strong>

    - <strong class="list-label">位置与速度</strong>:位于片下(off-chip)的DRAM/HBM中,是GPU上容量最大但访问速度最慢的内存空间。通过‘cudaMalloc‘等函数分配的内存就在这里。

    - <strong class="list-label">作用域</strong>:所有线程都可以访问,是实现不同线程块之间数据通信的唯一途径。

    - <strong class="list-label">生命周期</strong>:与整个应用程序的生命周期相同,除非被显式释放。

    - <strong class="list-label">缓存</strong>:对全局内存的访问会经过L1和L2缓存来提速。

5.  <strong class="critical-term">常量内存 (Constant Memory)</strong>

    - <strong class="list-label">位置与速度</strong>:物理上驻留在片下的设备内存中,但每个SM都有一个专用的只读常量缓存(Constant Cache),因此读取速度很快。

    - <strong class="list-label">作用域</strong>:对所有线程可见,但它是只读的。

    - <strong class="list-label">用途</strong>:主要用于存储在核函数执行期间不会改变的数据。当一个Warp(一组线程)中的多个线程需要访问同一个常量数据时,常量缓存可以实现广播(broadcast),一次性满足所有请求,避免了串行访问,从而提高效率。

6.  <strong class="critical-term">L1/L2 缓存 (Cache)</strong>

    - <strong class="list-label">位置</strong>:L1缓存位于每个SM内部,被该SM内的CUDA核心共享。L2缓存则被GPU上所有的SM共享。

    - <strong class="list-label">硬件类型</strong>:它们都是片上SRAM。

    - <strong class="list-label">作用</strong>:由系统自动控制,对程序员不完全透明。它们主要用于缓存对慢速内存(如全局内存和本地内存)的访问,以减少访问延迟。FlashAttention这类算法的核心思想就是尽可能利用L1/L2缓存来减少对HBM的读写。

#### CUDA通信机制知识补充

<span id="app-gpu-cuda"></span>

<strong class="critical-term">核心原则:异构计算</strong>

<div class="custom-block tip">

<p class="custom-block-title">例子</p>

异构计算是指结合<strong class="key-term">不同指令值和体系结构的设备(计算单元)</strong>之间的联合计算,比如 CPU、GPU、FPGA。在 CUDA 编程中主要就是指 CPU及内存 与 GPU和显存。这两者在<strong class="key-term">物理上独立,通过 PCle 总线连接通信</strong>。

</div>

<div class="custom-block tip">

<p class="custom-block-title">例子 · PCle 通信特点</p>

- <strong class="list-label">连接方式</strong>:PCIe采用<strong class="key-term">点对点的串行连接</strong>方式,每个设备都有自己的<strong class="key-term">专用连接</strong>,可以独享带宽,无需像 PCl 这种旧的并行总线那样共享同一个物理带宽。

- <strong class="list-label">拓扑结构</strong>:PCIe系统是一个<strong class="key-term">树状的层次结构</strong>,通常由代表CPU接口的根复合体、用于扩展端口的交换器(Switch)以及终端设备(Endpoint,如显卡、网卡、固态硬盘等)组成。

- <strong class="list-label">分层架构</strong>:PCIe协议采用<strong class="key-term">分层架构</strong>,分为事务层、数据链路层和物理层。

  - <strong class="list-label">事务层</strong>:负责生成和解析用于数据请求和响应的事务层数据包(TLP)。

  - <strong class="list-label">数据链路层</strong>:确保数据传输的完整性,负责错误检测和纠正。

  - <strong class="list-label">物理层</strong>:负责数据的串行化传输和链路管理。

</div>

<figure data-latex-placement="H">
<img src="/images/0a32cf5094.png" style="width:80.0%" alt="CUDA通信机制示意图" />
<figcaption>CUDA通信机制示意图</figcaption>
</figure>

CUDA 里由于 CPU 与 GPU 物理上的分离使得很有必要研究一下两者之间的通信。这也带来了GPU主要的性能瓶颈:通信与计算之间的不平衡,即<strong class="key-term">数据传输的速度远低于GPU内部的计算速度</strong>

CUDA中的通信可以分为两大类:

1.  <strong class="list-label">主机与设备(Host-Device)之间的通信</strong>:宏观层面的数据传输。

2.  <strong class="list-label">设备内部(Intra-Device)的通信</strong>:微观层面,线程之间的数据交换。

##### 主机与设备(Host-Device)通信

这是最基础的数据交互。典型的流程是:

1.  CPU将数据从主存复制到显存。

2.  GPU执行计算(Kernel Launch)。

3.  CPU将结果从显存复制回主存。

###### 同步(Synchronous)内存拷贝

<strong class="key-term">特性:</strong>

- <strong class="list-label">阻塞式(Blocking)</strong>:当CPU线程调用拷贝函数时,它会<strong class="key-term">暂停执行(阻塞)</strong>,直到数据拷贝<strong class="key-term">完全完成</strong>,然后才会继续执行后续的CPU代码。

- <strong class="list-label">简单易懂</strong>:逻辑清晰,不易出错。

- <strong class="list-label">性能瓶颈</strong>:CPU在等待期间完全闲置,无法与GPU的数据传输并行工作,浪费了宝贵的计算资源。

###### 异步(Asynchronous)内存拷贝与流(Streams)

为了克服同步拷贝的性能瓶颈,CUDA引入了<strong class="key-term">异步拷贝</strong>和<strong class="key-term">流(Stream)</strong>的概念。

<strong class="list-label">流(Stream):</strong> 可以理解为GPU上的一个<strong class="key-term">任务队列</strong>。你向一个流中添加一系列操作(如内存拷贝、核函数启动),GPU会按照你添加的顺序来执行这些操作。

<strong class="key-term">异步拷贝:</strong>

- <strong class="list-label">非阻塞式(Non-Blocking)</strong>:当CPU线程调用异步拷贝函数时,它只是将这个“拷贝任务”<strong class="key-term">提交到指定的流</strong>中,然后<strong class="key-term">立即返回</strong>,继续执行后续的CPU代码。CPU不会等待拷贝完成。

- <strong class="list-label">并行潜力</strong>:这使得CPU可以继续准备下一个数据块、启动其他任务,或者让数据传输与GPU计算<strong class="key-term">重叠(Overlap)</strong>执行,从而极大地提升整体效率。

<div class="custom-block tip">

<p class="custom-block-title">理解与提示</p>

<strong class="key-term">关键点:</strong>

- <strong class="list-label">异步操作必须指定一个流</strong>。

- <strong class="list-label">要使异步内存拷贝真正实现非阻塞,主机内存必须是页锁定(Page-Locked)或固定(Pinned)内存</strong>。因为操作系统可能会移动常规的可分页内存,而GPU的DMA引擎需要一个固定的物理地址。使用‘cudaMallocHost()‘或‘cudaHostAlloc()‘分配页锁定内存。

- <strong class="list-label">需要显式同步</strong>。既然CPU不等待,那你如何知道GPU任务何时完成？你需要使用同步函数,如‘cudaStreamSynchronize()‘(等待指定流完成)或‘cudaDeviceSynchronize()‘(等待设备上所有任务完成)。

</div>

##### 设备内部(Intra-Device)通信

当数据已经在GPU上后,成千上万的线程需要相互协作来完成计算。它们的通信方式和范围是严格分级的。

###### 线程块(Block)内部通信

这是最高效、最常用的设备内部通信方式。同一线程块内的线程可以<strong class="key-term">共享数据和进行同步</strong>。

<strong class="list-label">共享内存(Shared Memory)</strong>:

- <strong class="list-label">物理位置</strong>:位于GPU芯片上(On-chip),与L1缓存共享物理资源。

- <strong class="list-label">速度</strong>:访问速度<strong class="key-term">极快</strong>,远高于全局内存(Global Memory),几乎与寄存器(Register)访问速度相当。

- <strong class="list-label">生命周期</strong>:与线程块相同。当线程块开始执行时分配,执行完毕后释放。

- <strong class="list-label">作用域</strong>:仅对同一线程块内的所有线程可见。块A的线程无法访问块B的共享内存。

- <strong class="list-label">声明</strong>:使用 ‘\_\_shared\_\_‘ 限定符在核函数内部声明。

<strong class="list-label">同步原语(‘\_\_syncthreads\_\_()‘)</strong>:

- <strong class="list-label">功能</strong>:这是一个<strong class="key-term">栅栏(Barrier)</strong>。块内的所有线程执行到 ‘\_\_syncthreads\_\_()‘ 时必须停下来等待,直到<strong class="key-term">块内所有线程</strong>都到达这个点,然后才能继续执行后面的指令。

- <strong class="list-label">必要性</strong>:确保数据依赖性。例如,在线程A从共享内存读取数据之前,必须确保所有向该共享内存写入数据的线程(如线程B、C、D)都已经完成了写入操作。‘\_\_syncthreads\_\_()‘ 就是用来保证这一点的。

这是一个更复杂的问题。CUDA的编程模型<strong class="key-term">没有提供直接、高效的块间同步机制</strong>。不同块的执行顺序和并发性由硬件调度器决定,程序员无法假设块A一定在块B之前完成。

因此,块间通信通常通过<strong class="key-term">全局内存(Global Memory)</strong>间接实现:

1.  <strong class="list-label">分步执行(Separate Kernels)</strong>:

    - 这是最常用、最可靠的方法。

    - <strong class="list-label">流程</strong>:

      - a\. 启动一个核函数(Kernel 1),每个块完成其部分计算,并将中间结果写入全局内存。

      - b\. 核函数结束后,有一个隐式的全局同步。

      - c\. 启动第二个核函数(Kernel 2),读取Kernel 1产生的中间结果,进行下一阶段的计算。

    - <strong class="list-label">缺点</strong>:多次核函数启动会带来额外的开销。

2.  <strong class="list-label">原子操作(Atomic Operations)</strong>:

    - 当多个块中的线程需要更新<strong class="key-term">同一个全局内存地址</strong>时(例如,一个全局计数器),为了避免数据竞争,必须使用原子操作。

    - <strong class="list-label">函数</strong>:‘atomicAdd()‘, ‘atomicExch()‘, ‘atomicCAS()‘ 等。

    - <strong class="list-label">特性</strong>:保证操作的原子性(不会被中途打断),但可能会因为争用而导致线程串行化,影响性能。

3.  <strong class="list-label">持久化线程(Persistent Threads)与网格级同步(Grid-Level Sync)</strong> (高级)

    - 对于一些复杂的应用(如图计算、某些稀疏计算),可以启动一个覆盖整个设备的持久化网格(Grid),线程在循环中不断从全局队列中获取任务。

    - 较新的CUDA版本(计算能力7.x+)引入了<strong class="key-term">Cooperative Groups</strong>,提供了在整个Grid范围内同步的机制(‘grid.sync()‘)。这允许在<strong class="key-term">单个核函数内</strong>实现全局同步,避免了多次启动的开销。但这需要谨慎设计,且对硬件有要求。

### 参考文献与延伸阅读

####  · GPU 内存架构

<span id="read-8-1"></span>

- <https://www.cnblogs.com/ArsenalfanInECNU/p/18021724>

- 推荐参考本人整理的 infra 相关的笔记：<https://github.com/weiruihhh/aiinfra_notes>

### 训练与推理基础

<span id="app-infra-training-inference"></span>

#### 混合精度训练

<span id="sec-7-3"></span><span id="app-gpu-precision"></span> 到目前为止，在这个任务中，我们一直在使用FP32精度——所有模型参数和激活值都具有 torch.float32 数据类型。然而，现代 NVIDIA GPU包含专门的GPU核心（张量核心），用于以较低精度加速矩阵乘法。例如， NVIDIA A100规格表显示，其在FP32下的最大吞吐量为19.5 TFLOP /秒，而使用FP16（半精度浮点）或BF16（脑浮点）时，最大吞吐量显著提高至312 TFLOP /秒。因此，使用较低精度的数据类型有助于加快训练和推理速度。

然而，若简单地将我们的模型转换为低精度格式，可能会导致模型准确度降低。 在此情况下，混合精度训练（Mixed Precision Training）提供了一种解决方案。它允许我们使用较低精度（如FP16或BF16）来计算模型参数和激活值，同时保持FP32精度来计算梯度。这不仅提高了训练速度，还避免了精度丢失问题。

总结混合精度训练的特点:

朴素的训练默认使用32位浮点数（FP32），而混合精度训练引入了16位浮点数（FP16或BF16），主要带来两大好处：

1.  <strong class="critical-term">减少显存占用</strong>：模型参数、梯度和优化器状态占用的显存几乎减半。这意味着可以用同样的显存训练更大的模型，或者使用更大的批量（batch size）。

2.  <strong class="critical-term">加快训练速度</strong>：现代NVIDIA GPU内置了Tensor Cores，专门用于加速FP16/BF16的矩阵运算，其理论吞吐量远高于FP32。使用低精度计算能充分利用这部分硬件性能，带来显著的训练加速（通常能提升1.5倍至3倍）。

但是，混合精度训练也存在两大挑战：

1.  <strong class="critical-term">数据溢出</strong>：直接把所有东西都变成FP16是行不通的，因为它的数值范围（约$6e-5$ 到 $65504$）太小，非常容易出现<strong class="critical-term">上溢（Overflow）</strong>变成‘inf‘和<strong class="critical-term">下溢（Underflow）</strong>变成‘0‘，导致训练崩溃。

    - <strong class="note-label">下溢 (Underflow)</strong>：指一个数因为<strong class="key-term">太小</strong>（过于接近零），超出了当前数据类型能表示的最小精度范围，结果被强制舍入为<strong class="key-term">零</strong>。这就像用一把最小刻度是“毫米”的尺子去测量一粒灰尘，由于灰尘太小，你只能记录下它的长度是“0毫米”。

    - <strong class="note-label">上溢 (Overflow)</strong>：指一个数因为<strong class="key-term">太大</strong>，超出了当前数据类型能表示的最大范围，结果变成了一个特殊值——<strong class="key-term">无穷大 (‘inf‘)</strong>。这就像用一把30厘米的尺子去测量一张1米长的桌子，你只能说它的长度“超出了尺子的范围”。

2.  <strong class="critical-term">舍入误差</strong>：当网络模型的反向梯度很小时，一些FP32能够表示的数值可能不能满足FP16精度下的表示范围，导致被强行舍入，带来误差。比如 0.66….6（32位） 被舍入成 0.66..7（16位）。

<div class="custom-block tip">

<p class="custom-block-title">理解与提示</p>

<strong class="key-term">溢出</strong>是<strong class="key-term">范围问题</strong>，它是指数值超出了FP16表示的范围；而<strong class="key-term">舍入误差</strong>是<strong class="key-term">精度问题</strong>，它发生在数值可以被FP16表示范围呢，但精度不能够被FP16表示。

</div>

##### 浮点数知识回顾

定点数表示小数点位置是固定的，而浮点数则是用科学计数法来表示数值。

###### IEEE浮点表示

$V=(-1)^s\times M\times 2^{E}$

- <strong class="list-label">符号s：</strong>s=1代表负数，s=0代表正数

- <strong class="list-label">尾数M：</strong>尾数是浮点数的有效精度部分;

  e.g. $6.75=1.6875\times 2^2$, 1.6875就是尾数M。<strong class="key-term">尾数隐含以1开头，M=1+f,实际存的只有小数点后面的数</strong>

- <strong class="list-label">阶码E：</strong>决定数值的范围；

  <strong class="list-label">值得注意的是在规格化下</strong>：指数位会存在一个偏移，偏移的好处是为了方便数值大小的比较和避免0的不唯一表示。偏移量 $bias = 2^{指数位数-1} - 1$；比如8位指数的偏移量 $bias = 127$ 。

- 最后存储的指数 $E = e(真实指数值) + bias$

<figure data-latex-placement="H">
<img src="/images/1f2b5df054.png" style="width:80.0%" alt="IEEE浮点表示" />
<figcaption>IEEE浮点表示</figcaption>
</figure>

<strong class="list-label">半精度浮点数、单精度浮点数和双精度浮点数</strong>

- 半精度浮点数(FP16)包括1位符号位，5位指数，10位尾数。

- 单精度浮点数(FP32)包括1位符号位，8位指数，23位尾数。

- 双精度浮点数(FP64)包括1位符号位，11位指数，52位尾数。

<figure data-latex-placement="H">
<img src="/images/831b332d96.png" style="width:80.0%" alt="半精度浮点数、单精度浮点数和双精度浮点数" />
<figcaption>半精度浮点数、单精度浮点数和双精度浮点数</figcaption>
</figure>

###### 规格化、非规格化和特殊值

规格化：指数E既不是全0，也不是全1；$E=e-bias$,对于单精度全0→-126,全1→127。$bias=2^{k-1}-1$；尾数部分隐含1开头的情况；大部分都是这种情况。

非规格化：指数E全0；阶码$E=1-bias$;尾数不全0，没有隐含的1,$0.xxxxx \times 2^{-126}$为范围(单精度)。

特殊值：尾数和指数全为0时，代表±0

指数全为1，尾数全为0时，为±∞

当指数全为1，尾数非全0，为NaN

##### 混合精度训练的实现

###### 梯度缩放 (Loss Scaling):解决数据溢出问题

举个例子，在模型前向传播、计算损失使用FP16精度计算完了之后，某个权重计算得到的真实梯度是 ‘1e-6‘，如果不使用缩放还使用 FP16 去存储的话，那么因为‘1e-6‘已经不能被FP16半精度所表示，它就自动被视为0，这样就成了梯度消失，学习无效。

而缩放就是在计算得到梯度之后给它乘一个缩放因子，比如 65536，乘了之后的结果就是 FP16 可以表示的了。

<div class="custom-block tip">

<p class="custom-block-title">理解与提示</p>

在深度学习训练中，尤其是在模型后期趋于收敛时，计算出的梯度值经常会变得非常非常小，比如 ‘1e-6‘, ‘1e-7‘ 等，所以一般缩放因子都比较大，只针对<strong class="key-term">下溢</strong>而非上<strong class="key-term">溢出</strong>。

</div>

###### 权重备份（Weight Backup）：解决舍入误差

假设你有一个FP16格式的权重，值是 ‘0.9824‘。现在计算出一个梯度更新量，在FP32下是 ‘0.0010001‘。但在FP16下，这个更新量可能被舍入成 ‘0.001000‘。所以 ‘0.9824 + 0.001000‘ 在FP16下可能是 ‘0.9834‘。而最后一位的这个小更新就<strong class="critical-term">永久丢失</strong>了。一次两次没问题，成千上万次迭代后，大量的小更新都丢失了，模型就学偏了或不收敛了。

权重备份的思路就是<strong class="critical-term">把权重备份一个FP32精度的</strong>，每次计算梯度更新权重的时候就用FP32的来计算，得到新结果再转换成FP16，方便下一次计算梯度、激活等操作。

<figure data-latex-placement="H">
<img src="/images/7e8a8aa43d.png" style="width:80.0%" alt="权重备份" />
<figcaption>权重备份</figcaption>
</figure>

###### 精度累加（precision accumulated）：解决舍入误差

与权重备份类似，精度累加也是利用了FP32精度作为中介来解决舍入误差的问题，区别在于精度累加的过程是<strong class="critical-term">在计算矩阵乘法运算</strong>的时候，计算的结果用FP32保存减少舍入误差，然后再转成FP16格式进行下一步计算。

<figure data-latex-placement="H">
<img src="/images/8348769932.png" style="width:80.0%" alt="精度累加" />
<figcaption>精度累加</figcaption>
</figure>

<div class="custom-block tip">

<p class="custom-block-title">理解与提示</p>

事实上，在解决舍入误差的问题上，更新后的FP32主权重转换回FP16用于下一次计算时，会引入一次新的舍入误差。但这个误差是一次性的、非累积的。真正的关键在于，所有微小的梯度更新信息已经被精确地、永久地累积到了FP32的主权重上。我们避免了最致命的累积误差问题。

</div>

#### 计算量、显存与单位

<span id="app-units"></span> 配合[性能分析](/part-2/chapter-6#guide-ch-7)、[GPU 存储](/part-2/chapter-7#guide-ch-8)与[Scaling Law](/part-3/chapter-9#guide-ch-10)查阅。

##### 容量、速率与计算量

- <strong class="list-label">容量：</strong>1 byte = 8 bit；1 GB = $10^9$ byte，1 GiB = $2^{30}$ byte。

- <strong class="list-label">计算量：</strong>FLOPs 表示浮点运算次数；FLOP/s 表示每秒运算次数。缩写有歧义时结合上下文确认。

- <strong class="list-label">带宽与吞吐量：</strong>带宽常用 byte/s；吞吐量常用 token/s 或 sample/s。注明计时是否包含读取、同步和优化器更新。

<div class="key-formula">

$$
\text{吞吐量}=\frac{\text{实际处理的 token 或样本数量}}{\text{对应范围内的时间}}.
$$

</div>

##### 显存估算从哪些项开始

$P$ 个元素、每个占 $s$ byte，数据本体约占 $Ps$ byte。十亿个 FP16/BF16 元素约占 2 GB，即 1.86 GiB。

<strong class="note-label">训练显存：</strong>还包括梯度、优化器状态、激活、临时工作区、通信缓冲区及分配开销。各项精度随实现而变；参数字节数不能代表训练峰值。

##### 性能比较与规模估算

<strong class="list-label">性能比较：</strong>固定输入、精度、设备和正确性要求，再比较耗时与显存；分别记录理论峰值、实测吞吐量和利用率，区分硬件与算法影响。

<strong class="list-label">规模估算：</strong>稠密语言模型训练常粗估为 $C\approx6ND$：$N$ 为参数量，$D$ 为训练 token 数，$C$ 为计算量。使用前核对结构与计数假设。

### 参考文献与延伸阅读

<strong class="note-label">正文索引：</strong>[测量与显存快照](/part-2/chapter-6#guide-ch-7)、[GPU 存储与注意力优化](/part-2/chapter-7#guide-ch-8)、[Scaling Law 变量与预算](/part-3/chapter-9#guide-ch-10)。

## 信息论基础

<span id="app-information"></span><span id="sec-3-1"></span>

### 信息 (Information)

<span id="app-information-1"></span>

它用来<strong class="key-term">度量事件发生所消除的不确定性(Uncertainty)的大小</strong>。

1.  直观理解

    - “明天太阳会照常升起”:这句话包含的信息量很小。因为它几乎是必然发生的,没有消除我们什么不确定性。

    - “明天北京会下雪(假设在夏天)”:这句话包含的信息量极大。因为它是一个极小概率事件,一旦发生,就极大地消除了我们的不确定性。

2.  <strong class="list-label">量化定义:</strong>自信息 (Self-Information) 信息论使用“自信息”来量化单个事件发生时所提供的信息量。一个事件 x 的自信息量 I(x) 定义为:

    $I(x) = -\log_b(p(x))$

    - <strong class="list-label">p(x):</strong>事件 x 发生的概率。

    - $\log_b$:对数函数。底数 b 的选择决定了信息量的单位。

      - b=2:单位是 <strong class="key-term">比特 (bit)</strong>,这是最常用的单位,对应于二进制世界。

      - b=e:单位是 <strong class="key-term">奈特 (nat)</strong>。

      - b=10:单位是 <strong class="key-term">哈特利 (hartley)</strong>。

    - <strong class="list-label">负号:</strong>因为概率 p(x) 在 \[0, 1\] 之间,它的对数是小于等于0的。加上负号可以确保信息量是一个非负数,这符合我们的直觉。

3.  示例

    - <strong class="list-label">假设我们抛一枚均匀的硬币:</strong>

    - “正面朝上”的概率 p(正面) = 0.5。

      它提供的信息量 $I(正面) = -\log_2(0.5) = -\log_2(2^{-1}) = -(-1) = 1 bit$。

    - 同样,“反面朝上”提供的信息量也是 1 bit。

      这告诉我们,要确定一个硬币的正反面,我们需要 1 bit 的信息。

### 熵 (Entropy)

<span id="app-information-2"></span>

如果我们想衡量的是一个<strong class="key-term">系统(随机变量)的整体不确定性</strong>,而不是单个事件,那就要用到“熵”的概念。

1.  直观理解

    - 熵是<strong class="key-term">信息量的期望值(数学期望)</strong>。它衡量了一个随机变量所有可能结果的平均不确定性。

    - <strong class="key-term">一个系统越混乱、越不可预测,它的熵就越高</strong>。

    - <strong class="key-term">一个系统越稳定、越可预测,它的熵就越低</strong>。

2.  量化定义

    - 对于一个离散随机变量 X,它有多种可能的取值 $\{x_1, x_2, ..., x_n\}$,对应的概率为 $\{p(x_1), p(x_2), ..., p(x_n)\}$。那么,这个随机变量 X 的熵 H(X) 定义为:

    - $H(X) = E[I(X)] = \sum_{i=1}^{n} p(x_i) I(x_i) = -\sum_{i=1}^{n} p(x_i) \log_b(p(x_i))$

3.  示例

    - <strong class="list-label">比较两枚硬币的熵:</strong>

      - <strong class="list-label">均匀硬币 (Fair Coin):</strong>p(正面)=0.5, p(反面)=0.5

      - 不均匀硬币 (Biased Coin):p(正面)=0.9, p(反面)=0.1

      - 两面都是正面的硬币 (Two-headed Coin):p(正面)=1, p(反面)=0

    - <strong class="list-label">均匀硬币 (Fair Coin):</strong>p(正面)=0.5, p(反面)=0.5

      $H(X) = -[0.5 \times \log_2(0.5) + 0.5 \times \log_2(0.5)] = 1 bit$

    - 不均匀硬币 (Biased Coin):p(正面)=0.9, p(反面)=0.1

      $H(X) = -[0.9 \times \log_2(0.9) + 0.1 \times \log_2(0.1)] \approx 0.469 bit$

    - 两面都是正面的硬币 (Two-headed Coin):p(正面)=1, p(反面)=0

      $H(X) = -[1 \times \log_2(1) + 0 \times \log_2(0)] = 0 (约定 0 \log 0 = 0)$

### 熵的相关扩展概念

<span id="app-information-3"></span>

理解了熵之后,其他几个重要概念就很容易理解了,它们描述了多个随机变量之间的关系。

1.  <strong class="list-label">联合熵 (Joint Entropy)</strong>

    - 衡量<strong class="key-term">两个或多个随机变量共同</strong>的不确定性。对于两个变量 X 和 Y,其联合熵 H(X, Y) 为:

    - $H(X, Y) = -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2(p(x, y))$

    - 其中 p(x, y) 是 X=x 和 Y=y 同时发生的联合概率。

2.  <strong class="list-label">条件熵 (Conditional Entropy)</strong>

    - 在<strong class="key-term">已知一个随机变量 X 的情况下,另一个随机变量 Y 剩下的不确定性</strong>。记为 H(Y\|X)。

    - $H(Y|X) = \sum_{x \in X} p(x) H(Y|X=x)$

    - 它表示,知道了 X 之后,对 Y 的不确定性还剩下多少。

3.  <strong class="list-label">互信息 (Mutual Information)</strong>

    - <strong class="key-term">一个随机变量 X 的信息中,有多少是与另一个随机变量 Y 共享的</strong>。它衡量了两个变量之间的相关性。记为 I(X; Y)。

    - <strong class="note-label">直观理解</strong>:知道了 X 之后,Y 的不确定性减少了多少。

    - <strong class="list-label">计算公式</strong>:

      - $I(X; Y) = H(Y) - H(Y|X) (\KeyTerm{Y的总不确定性 - 知道X后Y剩下的不确定性})$

      - $I(X; Y) = H(X) - H(X|Y) (\KeyTerm{对称的})$

      - $I(X; Y) = H(X) + H(Y) - H(X, Y)$

Venn图关系 你可以把熵想象成一个集合,那么这些概念的关系就非常清晰了:

<figure data-latex-placement="H">
<img src="/images/77d0408750.png" style="width:80.0%" alt="信息论概念的Venn图关系" />
<figcaption>信息论概念的Venn图关系</figcaption>
</figure>

 

- 左边的圆圈是 H(X)

- 右边的圆圈是 H(Y)

- 两个圆圈的重叠部分是<strong class="key-term">互信息 I(X; Y)</strong>

- H(X) 中不重叠的部分是<strong class="key-term">条件熵 H(X\|Y)</strong>

- H(Y) 中不重叠的部分是<strong class="key-term">条件熵 H(Y\|X)</strong>

- 整个两个圆圈覆盖的区域是<strong class="key-term">联合熵 H(X, Y)</strong>

### 一个生动的比喻:猜数字游戏

<span id="app-information-4"></span>

这个比喻可以把所有概念串起来。

- <strong class="list-label">游戏</strong>:我从1到8之间想一个数字,你来猜。

- <strong class="list-label">熵 H(X)</strong>:在游戏开始前,这个数字是什么？你完全不知道,有8种可能性,每种可能性概率为1/8。这个系统的总不确定性是 $H(X) = -\sum_{i=1}^{8} \frac{1}{8}\log_2(\frac{1}{8}) = \log_2(8) = 3 bits$ 。这恰好是你用“二分法”猜中这个数字所需的最少问题数(例如:“比4大吗？”“比6大吗？”“是7吗？”)。

- <strong class="list-label">信息 I(x)</strong>:我告诉你答案是“5”。这个具体事件提供的信息量是 $I("5") = -\log_2(1/8) = 3 bits$ 。一旦你知道了这个信息,所有不确定性都消除了。

- <strong class="list-label">互信息 I(X;Y)</strong>:现在,我不直接告诉你答案,而是给你一个提示 <strong class="key-term">Y</strong>:“这个数字是奇数”。

  - 这个提示 <strong class="key-term">Y</strong> 本身也有不确定性(可能是奇数或偶数),它的熵是 <strong class="key-term">H(Y)=1 bit</strong>。

  - 这个提示 <strong class="key-term">Y</strong> 给你提供了多少关于 <strong class="key-term">X</strong> 的信息？这就是互信息。知道了它是奇数,可能性从1,2,3,4,5,6,7,8缩小到了1,3,5,7。不确定性大大降低。

- <strong class="list-label">条件熵 H(X\|Y)</strong>:你知道了“数字是奇数” <strong class="key-term">(Y)</strong> 之后,对数字 <strong class="key-term">(X)</strong> 还剩下多少不确定性？

  - 现在只剩下4种可能性1,3,5,7,每种概率为1/4。

  - 剩下的不确定性(条件熵)是 $H(X|Y=\text{"奇数"}) = -\sum_{i \in \{1,3,5,7\}} \frac{1}{4}\log_2(\frac{1}{4}) = \log_2(4) = 2 bits$。

  - 这意味你还需要问2个问题才能猜到。

- <strong class="list-label">关系验证</strong>:

  - $I(X;Y) = H(X) - H(X|Y) = 3 - 2 = 1 bit。$

  - 这说明,“数字是奇数”这个提示,为你提供了 1 bit 的信息。

| <strong>中文名称</strong> | <strong>描述</strong> | <strong>关键点</strong> |
|:--:|:---|:---|
| <strong>自信息</strong> | 度量<strong>单个具体事件</strong>发生所消除的不确定性。 | 概率越小,信息量越大。 |
| <strong>熵</strong> | 度量<strong>整个系统(随机变量)</strong>的平均不确定性。 | 信息量的数学期望。系统越混乱,熵越高。 |
| <strong>联合熵</strong> | 度量<strong>两个或多个系统</strong>共同的不确定性。 | 把多个变量看成一个整体。 |
| <strong>条件熵</strong> | 在<strong>已知一个变量</strong>后,另一个变量<strong>剩下</strong>的不确定性。 | $H(Y\|X)$,知道X后Y还有多乱。 |
| <strong>互信息</strong> | 两个变量<strong>共享</strong>的信息量,即相关性程度。 | $I(X;Y)$,知道X能帮我们消除Y多少不确定性。 |

## 正则表达式与文本匹配

<span id="app-regex"></span> 本附录供阅读[分词器与 BPE](/part-1/chapter-1#guide-ch-1)中的预分词部分时查阅。

### 概念与常用符号

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 正则表达式</p>

正则表达式（Regular Expression,常缩写为regex或regexp）是一个强大的文本模式匹配工具。它本质上是一种用特殊符号编写的“规则字符串”,可以用来<strong class="key-term">查找、替换、分割或验证任何符合该规则的文本</strong>。可以理解为ctrl+f的超级加强版。

</div>

<table>
<caption>正则表达式符号表<span id="tab-regex_symbols"></span></caption>
<thead>
<tr>
<th style="text-align: left;">

<strong>类型</strong>

</th>

<th style="text-align: left;">

<strong>符号/语法</strong>

</th>

<th style="text-align: left;">

<strong>解释说明</strong>

</th>

<th style="text-align: left;">

<strong>示例</strong>

</th>

</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">

普通字符

</td>

<td style="text-align: left;">

<code>a</code>, <code>b</code>, <code>1</code>, <code>2</code>

</td>

<td style="text-align: left;">

匹配它们自身。

</td>

<td style="text-align: left;">

<code>cat</code> 会精确匹配字符串 "cat"。

</td>

</tr>
<tr>
<td style="text-align: left;">

元字符（任意字符）

</td>

<td style="text-align: left;">

<code>.</code>

</td>

<td style="text-align: left;">

匹配<strong>除了换行符以外</strong>的任意单个字符。

</td>

<td style="text-align: left;">

<code>c.t</code> 会匹配 "cat", "cot", "c_t" 等。

</td>

</tr>
<tr>
<td rowspan="3" style="text-align: left;">

元字符（重复次数）

</td>

<td style="text-align: left;">

<code>*</code>

</td>

<td style="text-align: left;">

匹配前面的元素 <strong>0次或多次</strong>。

</td>

<td style="text-align: left;">

<code>ca*t</code> 会匹配 "ct", "cat", "caaat"。

</td>

</tr>
<tr>
<td style="text-align: left;">

<code>+</code>

</td>

<td style="text-align: left;">

匹配前面的元素 <strong>1次或多次</strong>。

</td>

<td style="text-align: left;">

<code>ca+t</code> 会匹配 "cat", "caaat",但<strong>不匹配</strong> "ct"。

</td>

</tr>
<tr>
<td style="text-align: left;">

<code>?</code>

</td>

<td style="text-align: left;">

匹配前面的元素 <strong>0次或1次</strong>。

</td>

<td style="text-align: left;">

<code>colou?r</code> 会匹配 "color" 和 "colour"。

</td>

</tr>
<tr>
<td rowspan="2" style="text-align: left;">

字符集

</td>

<td style="text-align: left;">

<code>[...]</code>

</td>

<td style="text-align: left;">

匹配方括号内的<strong>任意一个</strong>字符。

</td>

<td style="text-align: left;">

<code>c[ao]t</code> 只会匹配 "cat" 和 "cot"。

</td>

</tr>
<tr>
<td style="text-align: left;">

<code>[^...]</code>

</td>

<td style="text-align: left;">

匹配<strong>不在</strong>方括号内的任意一个字符。

</td>

<td style="text-align: left;">

<code>[^0-9]</code> 会匹配任何非数字字符。

</td>

</tr>
<tr>
<td rowspan="2" style="text-align: left;">

分组与或

</td>

<td style="text-align: left;">

<code>(...)</code>

</td>

<td style="text-align: left;">

将括号内的内容视为一个整体,可以对整体做重复。

</td>

<td style="text-align: left;">

<code>(ab)+</code> 会匹配 "ab", "abab", "ababab"。

</td>

</tr>
<tr>
<td style="text-align: left;">

<code>|</code>

</td>

<td style="text-align: left;">

表示"或"（OR）逻辑。

</td>

<td style="text-align: left;">

<code>cat</code>dog| 会匹配 "cat" 或者 "dog"。

</td>

</tr>
<tr>
<td rowspan="3" style="text-align: left;">

预定义字符类

</td>

<td style="text-align: left;">

<code>\d</code>

</td>

<td style="text-align: left;">

匹配任意一个<strong>数字</strong> (Digit),等同于 <code>[0-9]</code>。

</td>

<td style="text-align: left;">

<code>\d\d\d</code> 会匹配 "123", "987"。

</td>

</tr>
<tr>
<td style="text-align: left;">

<code>\w</code>

</td>

<td style="text-align: left;">

匹配任意一个<strong>单词字符</strong>,包括字母、数字、下划线。

</td>

<td style="text-align: left;">

<code>\w+</code> 会匹配一个完整的单词或数字。

</td>

</tr>
<tr>
<td style="text-align: left;">

<code>\s</code>

</td>

<td style="text-align: left;">

匹配任意一个<strong>空白字符</strong>,包括空格、制表符、换行符。

</td>

<td style="text-align: left;">



</td>

</tr>
</tbody>
</table>

### Python 使用示例

<div class="custom-block tip">

<p class="custom-block-title">例子 · 在python代码中使用正则表达式</p>

re.findall和re.finditer的区别:

re.findall返回所有匹配的子字符串,返回一个列表。

re.finditer则是<strong class="key-term">惰性的</strong>,返回一个迭代器,每次只返回一个匹配的子字符串,需要手动调用next()方法来获取下一个匹配的子字符串。正因如此,无论文本有多大,匹配项有多少,<strong class="key-term">内存占用都极低</strong>,因为它一次只处理一个匹配项。这是处理大文件的唯一可行方法。讲义中也推荐使用re.finditer。

</div>

```python
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    re.findall(PAT, "some text that i'll pre-tokenize")
    >>> ['some', 'text', 'that', 'i', "'ll", 'pre', '-', 'tokenize']
```

### 参考文献与延伸阅读

正则表达式相关知识: <https://www.runoob.com/regexp/regexp-intro.html>
