---
outline: [2, 3]
---

# 第 12 章 · 策略优化基础：从策略梯度到 PPO

<span id="guide-ch-12-policy"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>完成[数学任务、SFT 与专家迭代](/part-5/chapter-11#guide-ch-12)；理解概率、期望与梯度。

<strong>准备工作：</strong>沿用数学任务中的问题、回答与评分结果，区分完整回答和 token 两个层次。

<strong>本章任务：</strong>建立语言模型的强化学习表述，理解策略梯度、价值函数、优势与 GAE，再通过重要性采样、TRPO 和 PPO 理解如何利用旧策略数据并控制更新幅度。

</div>

## 12.1 将语言模型训练写成强化学习问题

前面的专家迭代通过筛选样本后做 SFT 来改进模型。下面转向利用奖励构造策略更新，先统一语言模型与强化学习的符号。

- <strong class="list-label">状态 $s_t$：</strong>输入问题与已经生成的回答前缀。

- <strong class="list-label">动作 $a_t$：</strong>当前步骤生成的下一个 token。

- <strong class="list-label">策略 $\pi_\theta(a_t\mid s_t)$：</strong>语言模型给出的下一个 token 的条件概率分布。

- <strong class="list-label">轨迹与奖励：</strong>一次完整生成形成回答；评分函数评价回答，训练时需要把这个反馈用于各生成位置的更新。

<strong class="note-label">两个层次：</strong>评分与组内比较通常围绕完整回答展开，模型的 log-prob、response mask 和损失计算则落实到 token 位置。阅读后面的公式时，需要确认下标指向回答还是 token。

<strong class="note-label">三种策略：</strong>当前策略用于计算梯度；旧策略是生成当前批次回答时的策略快照；参考策略用于 KL 正则。旧策略和参考策略承担不同职责，后文实现时应分别记录。

## 12.2 策略梯度与优势函数

<strong class="note-label">延伸阅读：</strong>[见本篇末资料 \[3\]。](/part-5/chapter-13#read-12-3)

无论是朴素梯度策略,TRPO还是比较流行的PPO、GRPO、DPO,它们都有一个共同点,那就是它们的关注点是<strong class="critical-term">策略函数 $\pi_\theta(a|s)$</strong>,而不是像DQN这种算法先去学一个价值函数 $Q(s,a)$ 再反推策略。

### 12.2.1 价值函数、基线与优势

- <strong class="list-label">$V^\pi(s)$</strong>:状态价值函数,表示从状态 s 开始,遵循策略 π 所能获得的期望累积回报

- <strong class="list-label">$Q^\pi(s,a)$</strong>:动作价值函数,表示在状态 s 采取动作 a,之后遵循策略 π 所能获得的期望累积回报

- <strong class="list-label">$A^\pi(s,a)$</strong>:优势函数,定义为动作价值函数与状态价值函数之差:$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$。(不过在TRPO、PPO这种关注策略的方法里我们只能看到轨迹回报,无法得到Q(s,a) 和V(s)这些<strong class="key-term">期望值</strong>,因此采用的方式是<strong class="key-term">GAE</strong>来得到A)

将 $Q^\pi(s,a)$ 的贝尔曼方程代入优势函数定义:

$\begin{aligned} A^\pi(s,a) &= Q^\pi(s,a) - V^\pi(s) \\ &= \mathbb{E}_{s'}[r(s,a,s') + \gamma V^\pi(s')] - V^\pi(s) \\ &= \mathbb{E}_{s'}[r(s,a,s') + \gamma V^\pi(s') - V^\pi(s)] \end{aligned}$

这个形式揭示了优势函数的本质:<strong class="critical-term">它衡量的是采取动作 a 后获得的即时奖励加上未来价值,相对于当前状态平均价值的增量</strong>。

一个重要的数学性质是,优势函数在策略 π 下的期望为零:

$\mathbb{E}_{a \sim \pi(\cdot|s)} [A^\pi(s,a)] = \sum_{a} \pi(a|s) A^\pi(s,a) = \sum_{a} \pi(a|s) [Q^\pi(s,a) - V^\pi(s)]$

$= \sum_{a} \pi(a|s) Q^\pi(s,a) - V^\pi(s) \sum_{a} \pi(a|s) = V^\pi(s) - V^\pi(s) = 0$

这个性质表明:<strong class="key-term">优势函数是一个"零中心化"的量,它只关注相对好坏,而非绝对价值</strong>。

#### 12.2.1.1 直观理解

可以在以股市买股票打比方(×)

- <strong class="list-label">$V^\pi(s)$</strong>:是你随便买股票(按照当前策略随机选择方向)最终赚到的钱。

- <strong class="list-label">$Q^\pi(s,a)$</strong>:是你按照自身的判断只买特定的股票(比如买茅台)最终赚到的钱。

- <strong class="list-label">$A^\pi(s,a)$</strong>:就是你自己凭借自己判断买茅台比随便选股票多赚到的钱。

所以当 $A^\pi(s,a) > 0$:这个动作比平均水平好,应该增加其概率,反之则减少。

### 12.2.2 Vanilla Policy Gradient(朴素策略梯度)

这是所有策略梯度算法的基础。它的核心思想非常简单:<strong class="key-term">如果一个动作带来了高回报,就通过梯度上升增加该动作的概率。</strong>

#### 12.2.2.1 核心定义与符号

- <strong class="list-label">轨迹</strong>: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$。

- <strong class="list-label">轨迹回报</strong>: $R(\tau) = \sum_{t=0}^T \gamma^t r_t$。

- <strong class="list-label">目标函数</strong>: 最大化期望回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$。

- <strong class="list-label">$\theta \in \mathbb{R}^n$</strong>: 模型的参数向量(例如神经网络的权重)。

- <strong class="list-label">x</strong>: 随机变量(在 RL 中对应状态-动作对 (s,a))。

#### 12.2.2.2 数学原理推导

我们的目标是<strong class="key-term">计算梯度</strong> $\nabla_\theta J(\theta)$。 $J(\theta) = \int P(\tau|\theta) R(\tau) \, d\tau$

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

因为$R(\tau)$奖励值是确定好的,然后$P(\tau|\theta)=P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t)$,P()是和环境交互的反馈,我们能调整的也就是策略函数 $\pi_\theta(a|s)$这一项。更具体一点,也就是修改网络权重参数$\theta$,使得$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$,所以求梯度 $\nabla_\theta J(\theta)$就是必经之路。

</div>

其中 $P(\tau|\theta)$ 是轨迹发生的概率。对 $\theta$ 求导: $\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau|\theta) R(\tau) \, d\tau$ 这里运用著名的 <strong class="key-term">Log-Derivative Trick (对数导数技巧)</strong>:

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · Log-Derivative Trick (对数导数技巧)</p>

$\nabla \log x = \frac{\nabla x}{x}$,所以 $\nabla x = x \nabla \log x$。(说白了也就是链式法则移个项)

</div>

代入上式有:

$$
\nabla_\theta J(\theta) = \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) \, d\tau
$$

<div class="key-formula">

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau|\theta) R(\tau) \right]
$$

</div>

展开轨迹概率的对数(状态转移概率与 $\theta$ 无关,求导为0,只剩下策略项): $\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)$

最终得到 <strong class="key-term">Vanilla PG 的梯度公式</strong>: $\nabla_\theta J(\theta) = \mathbb{E}_{t} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot {A}_t \right]$

> <strong class="note-label">注:</strong>实际中通常用优势函数 ${A}_t$ 替换回报 $R(\tau)$ 以降低方差

<figure>
<img src="/images/e9fcf89ff9.png" style="width:100.0%" alt="Vanilla PG 的梯度公式" />
<figcaption>Vanilla PG 的梯度公式</figcaption>
</figure>

#### 12.2.2.3 直观理解

- <strong class="note-label">直观理解</strong>: $\nabla_\theta \log \pi_\theta$ 表示参数梯度方向,${A}_t$ 是标量权重,和$\alpha$共同组成步长大小。如果 ${A}_t$ 很大(表示当前的动作比平均动作带来的收益要好),梯度就很大,参数就大幅更新以增加该动作概率。

- <strong class="list-label">致命缺陷</strong>:

  1.  <strong class="list-label">步长 $\alpha$ 难以确定</strong>: 步长太小,训练极慢；步长太大,一次糟糕的更新会让策略参数 $\theta$ 飞到一个极差的区域。由于数据是根据当前策略采样的,策略变差后采样的数据更差,导致<strong class="key-term">无法恢复</strong>。

  2.  <strong class="list-label">采样效率低</strong>: 由于是<strong class="key-term">在线的(On-Policy)</strong>,每次更新完策略,旧数据就作废了。

## 12.3 GAE

<strong class="key-term">GAE (Generalized Advantage Estimation)</strong> 是现代强化学习(尤其是 Actor-Critic 架构)中处理<strong class="key-term">偏差-方差权衡</strong>,它的核心贡献在于它提供了一种数学上优雅的方法,通过调节参数 λ,在无偏的高方差估计(Monte-Carlo)和有偏的低方差估计(TD)之间找到trade-off平衡。

### 12.3.1 省流版

在PPO这种基于策略的on-policy算法里,我们采样的最终结果或者说是样本是一条条完整的轨迹及其奖励/回报,我们知道优势函数$A_t$的定义是 $A_t = Q^\pi(s,a) - V^\pi(s)$ ,用样本去估计的结果就是 $A_t = G_t - V^\pi(s)$ ,$G_t$就表示实际采样的结果,那么这个缺陷就比较明显,因为即使是同一策略,不同的轨迹样本之间的差别也会很大,这就导致了方差变大。因此GAE的想法是从TD误差上做文章,TD误差简单说就是把$Q^\pi(s,a)$表示成$r(s,a,s') + \gamma V^\pi(s')$这种开展一步或两步或n步的形式,这样方差就不像轨迹那样大了,因为我只关注某几步小变化。具体来讲,GAE 则把同一条轨迹的信息拆成多尺度的 TD 信号,并用 λ 平滑融合,从而得到好的优势函数$A_t$

### 12.3.2 核心定义与符号体系

- <strong class="list-label">$\lambda \in [0, 1]$</strong>: GAE 平滑参数,控制偏差与方差的权衡。

- <strong class="list-label">$\hat{A}_t^{\text{GAE}}$</strong>: 广义优势估计量。

- <strong class="list-label">$\delta_t^V$</strong>: TD 误差 。

### 12.3.3 问题背景

$A_\pi(s_t, a_t) = \mathbb{E}{s{t+1}}\left[ r_t + \gamma V_\pi(s_{t+1}) \right] - V_\pi(s_t)$

但在实际训练中,我们无法获知真实的 $V_\pi$(只能用神经网络近似 $V_\phi$),也不知道未来的确切奖励。因此我们需要<strong class="key-term">估计优势函数</strong>。

### 12.3.4 数学原理推导

#### 12.3.4.1 Monte Carlo Advantage

$A_t = G_t - V(s_t)$

- 用轨迹的<strong class="list-label">完整回报</strong> $G_t$

- <strong class="list-label">无偏</strong>(因为毕竟是真实值),但由于是轨迹而不是某几个点,导致方差较大

#### 12.3.4.2 TD Error ($\delta$)

我们定义 $V_\phi$ 下的 TD Error 为 $\delta_t^V$。这是最基础的单步优势估计: $\delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

<strong class="note-label">注意:</strong>这里 $\delta_t^V$ 其实就是 $A(s_t, a_t)$ 的一个<strong class="key-term">有偏估计</strong>(因为 $V_\phi$ 不是真实的 $V_\pi$),但它的方差很低,因为它只依赖一步真实的 $r_t$。

</div>

#### 12.3.4.3 展开多步估计

我们可以构造一系列的优势估计量 $A^{(k)}_t$,代表向前看 k 步:

- <strong class="list-label">1-step (k=1):</strong> $\hat{A}t^{(1)} = \delta_t^V = r_t + \gamma V\phi(s_{t+1}) - V_\phi(s_t)$

- <strong class="list-label">2-step (k=2):</strong> $\hat{A}_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V_\phi(s_{t+2}) - V_\phi(s_t)$

- <strong class="list-label">k-step:</strong> $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = -V_\phi(s_t) + \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V_\phi(s_{t+k})$

- <strong class="list-label">$\infty$-step (其实就等价Monte Carlo):</strong> $\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \left(\sum_{l=0}^\infty \gamma^l r_{t+l}\right) - V_\phi(s_t)$

  <strong class="key-term">(低偏差,高方差 - 因为累积了每一步环境的随机性)</strong>

#### 12.3.4.4 GAE 的核心:指数加权平均

GAE 的优点在于,他不选择特定的 k,而是将所有可能的 k-step 估计量进行<strong class="key-term">指数加权平均</strong>。权重由参数 $\lambda$ 控制。

定义 GAE 为: $\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1-\lambda) \left( \hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \dots \right)$

这是一个几何级数求和。推导后的最终形态:

$\hat{A}t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$

<figure>
<img src="/images/086dbedfe7.png" style="width:100.0%" alt="GAE" />
<figcaption>GAE</figcaption>
</figure>

#### 12.3.4.5 工程递归计算公式

在代码实现中,我们不会计算无穷级数,而是使用递归形式,从轨迹的最后一步向前计算:

<span class="key-formula">$\hat{A}_t^{\text{GAE}} = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}$</span>

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$。

采样的轨迹会提供一系列奖励r,然后V(s)可以通过神经网络得到。

### 12.3.5 直觉理解

#### 12.3.5.1 $\lambda$ 作为“置信度滑块”

你可以把 $\lambda$ 想象成一个调节我们对“<strong class="key-term">Critic 网络预测能力 ($V_\phi$)</strong>” vs <strong class="key-term">“现实世界反馈 ($r_t$)”</strong> 信任程度的滑块。

- <strong class="list-label">当 $\lambda = 0$ (TD):</strong> $\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

  - <strong class="list-label">物理意义</strong>:目光短浅。我们只相信当前这一步的奖励 $r_t$,对于未来,我们完全依赖 Critic 的预测 $V(s_{t+1})$。

  - <strong class="list-label">后果</strong>:方差最小(只引入了一步随机性),但偏差最大(如果 Critic 没训练好,估计就全错了)。

- <strong class="list-label">当 $\lambda = 1$ (Monte Carlo):</strong> $\hat{A}_t = \sum \gamma^l \delta_{t+l} = \text{Return} - V(s_t)$

  - <strong class="list-label">物理意义</strong>:实事求是。我们完全不信任 Critic 对未来的预测,我们一直等到 episode 结束,把所有实际拿到的奖励加起来。

  - <strong class="list-label">后果</strong>:偏差最小(基于真实数据),但方差最大(环境中的每一点风吹草动都会累积到回报中,导致训练震荡)。

- <strong class="list-label">当 $0 < \lambda < 1$ (GAE):</strong> 我们构建了一个<strong class="key-term">衰减窗口</strong>。我们利用近期的真实奖励来修正 Critic 的预测,同时利用 Critic 的预测来平滑远期的随机性。

<strong class="note-label">延伸阅读：</strong>信号处理与频率视角见[对应补充节](/part-5/chapter-13#supp-gae)。

## 12.4 从重要性采样到信任域优化

<strong class="note-label">延伸阅读：</strong>[见本篇末资料 \[2\]。](/part-5/chapter-13#read-12-2)

重要性采样(Importance Sampling)也是<strong class="critical-term">蒙特卡洛方法</strong>的一种<strong class="key-term">在线策略(On-policy)</strong>与<strong class="key-term">离线策略(Off-policy)</strong>上有重要作用。

重要性采样解决的核心问题是:<strong class="key-term">当我们无法直接从目标分布中采样,或者为了降低方差而故意从另一个分布中采样时,如何无偏地估计原分布下的期望值</strong>。

### 12.4.1 核心定义与符号体系

- $x \in \mathcal{X}$:随机变量(例如:强化学习中的轨迹、图像生成中的噪声向量)。

- $p(x)$:<strong class="key-term">目标分布</strong>。这是我们真正关心的分布,我们希望计算关于它的统计量。

- $f(x)$:我们关心的<strong class="key-term">目标函数</strong>(例如:奖励函数、损失函数)。我们想计算的是期望 $\mathbb{E}_{x \sim p}[f(x)]$。

- $q(x)$:<strong class="key-term">提议分布</strong>。这是我们实际用来进行采样的分布(通常 $q(x)$ 比 $p(x)$ 更容易采样,或者能采到更多“重要”样本)。

- $w(x)$:<strong class="key-term">重要性权重</strong>,一般定义为$w(x) = \frac{p(x)}{q(x)})$。

### 12.4.2 数学原理推导

我们的目标是计算函数 $f(x)$ 在目标分布 $p(x)$ 下的期望值:$I = \mathbb{E}_{x \sim p}[f(x)]$

#### 12.4.2.1 积分形式展开

根据期望的定义,将其写为积分形式: $I = \int_{\mathcal{X}} f(x) p(x) \, dx$

#### 12.4.2.2 引入提议分布

假设 $q(x)$ 是一个已知且易于采样的分布。我们在积分内部同时乘以并除以 $q(x)$。 <strong class="key-term">数学公理约束</strong>:为了保证这一步合法,必须满足<strong class="key-term">绝对连续性</strong>,即对于任意 x,若 $p(x)f(x) \neq 0$,则必须有 $q(x) > 0$。简言之,<strong class="key-term">q 的支撑集(Support)必须覆盖 p 的支撑集</strong>。

> 只要 p(x) 认为某件事有可能发生(概率 \> 0),那么 q(x) 也必须认为这件事有可能发生(概率 \> 0)。

$I = \int_{\mathcal{X}} f(x) \frac{p(x)}{q(x)} q(x) \, dx$

#### 12.4.2.3 重组为新的期望

我们将 $\frac{p(x)}{q(x)}$ 视为样本 x 的权重,记为 $w(x)$。此时,积分项 $q(x) dx$ 代表了我们在 q 分布下的概率测度。 于是,原积分转化为在分布 $q(x)$ 下的新期望:

$I = \mathbb{E}_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right] = \mathbb{E}_{x \sim q} [ f(x) w(x) ]$

#### 12.4.2.4 蒙特卡洛估计

在实际计算中,我们无法计算解析解积分,只能通过从 q(x) 中采样 N 个样本 $\{x_i\}_{i=1}^N$ 来进行离散近似:

$\hat{I}{IS} = \frac{1}{N} \sum{i=1}^N f(x_i) \frac{p(x_i)}{q(x_i)}$

这就是<strong class="key-term">重要性采样估计量</strong>。它证明了:<strong class="key-term">我们可以通过采样自己提出的提议分布 q,并通过权重 w 修正,从而得到 p 分布下的无偏估计</strong>。

#### 12.4.2.5 缺陷:方差分析

方差是重要性采样分析的重要缺陷。 估计量的方差为: $\text{Var}_{x \sim q}[\hat{I}_{IS}] = \frac{1}{N} \text{Var}_{x \sim q} \left( f(x) \frac{p(x)}{q(x)} \right)$ 如果 $q(x)$ 在某些 $p(x)f(x)$ 很大的区域取值很小(即 $q(x)$ 是轻尾的),那么权重 $w(x) = p(x)/q(x)$ 会爆炸式增长,导致估计量的方差趋于无穷大。这就是为什么选择合适的 $q(x)$ 至关重要。

### 12.4.3 举例理解

#### 12.4.3.1 任务目标

假设 x 服从标准正态分布 $x \sim \mathcal{N}(0, 1)$。 我们想计算 x 落在 3 之外的概率: $P(x > 3) = \mathbb{E}_{x \sim p} [\mathbb{I}(x > 3)]$

这里:

- <strong class="list-label">目标分布</strong> $p(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$

- <strong class="list-label">目标函数</strong> $f(x) = \mathbb{I}(x > 3)$。这是一个<strong class="key-term">指示函数</strong>:当 $x > 3$ 时为 1,否则为 0。

- <strong class="list-label">真实值</strong>:根据统计学表,这个概率约为 <strong class="key-term">0.00135</strong>(即千分之 1.35)。

#### 12.4.3.2 朴素蒙特卡洛采样

如果我们直接老老实实地从 $p(x)$(标准正态分布)中采样:

因为绝大多数 x 都会落在 \[-2, 2\] 之间。所以采了 100 次,很可能所有样本都小于 3,结果 $\hat{I} = 0$。

而且<strong class="key-term">方差极大</strong>:你要么算出来是 0,要么因为偶尔遇到一个样本而剧烈波动。

#### 12.4.3.3 重要性采样

为了解决“采不到”的问题,我们故意从一个更容易产生 $x > 3$ 的分布中采样,比如均值为 4 的正态分布。

1.  <strong class="list-label">设计提议分布 q(x)</strong>

    设 $q(x) \sim \mathcal{N}(4, 1)$。

    - 在这个分布里,x 很容易大于 3(因为中心就在 4)。

    - 它的支撑集是 $\mathbb{R}$,覆盖了 $p(x)$ 的支撑集,满足绝对连续性。

2.  <strong class="list-label">推导权重 $w(x)$</strong>

    对于任意采样到的样本 x,我们需要计算权重 $w(x) = \frac{p(x)}{q(x)}$。

    $\begin{aligned} p(x) &= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right) \\ q(x) &= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-4)^2}{2}\right) \end{aligned}$

    代入计算权重(常数项约掉): $w(x) = \frac{\exp(-x^2/2)}{\exp(-(x-4)^2/2)} = \exp\left( -\frac{x^2}{2} + \frac{(x-4)^2}{2} \right)$

    展开指数部分: $-\frac{x^2}{2} + \frac{x^2 - 8x + 16}{2} = \frac{-8x + 16}{2} = -4x + 8$ 所以,权重的解析式极为简洁: $\mathbf{w(x) = e^{8 - 4x}}$

3.  <strong class="list-label">实际计算流程</strong>

    假设我们进行一次采样:

    1.  <strong class="list-label">采样</strong>:从 $q(x) \sim \mathcal{N}(4, 1)$ 中抽取一个样本。假设抽到了 $x = 3.5$。

    2.  <strong class="list-label">计算目标函数</strong>:

        $f(3.5) = \mathbb{I}(3.5 > 3) = 1$

    3.  <strong class="list-label">计算权重</strong>: $w(3.5) = e^{8 - 4(3.5)} = e^{8 - 14} = e^{-6} \approx 0.00248$

    4.  <strong class="list-label">最终贡献</strong>: 这个样本对期望的贡献是 $1 \times 0.00248 = 0.00248$。

    <strong class="list-label">结果分析</strong>: 请注意,这个单次样本的估计值 0.00248 已经非常接近真实值 0.00135 了。 相比之下,朴素方法采样一次得到的通常是 0。<strong class="key-term">IS 极大地提高了采样效率</strong>。

### 12.4.4 旧策略数据与概率比率

在PPO的实际训练中,我们手里只有<strong class="key-term">旧策略 $\pi_{\theta_{old}}$</strong> 采样的数据。为了计算<strong class="key-term">新策略 $\pi_\theta$</strong> 的期望,我们必须使用<strong class="key-term">重要性采样 (IS)</strong> 变换。

$L^{CPI}(\theta) = \mathbb{E}_{t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right]$

这就是 TRPO/PPO 中那个比率 $\frac{\pi_\theta}{\pi_{\theta_{old}}}$ 的由来。它本质上是在<strong class="key-term">修正用旧策略数据评估新策略表现时的概率偏差</strong>。

### 12.4.5 TRPO

TRPO 是为了解决 Vanilla PG 的步长问题而提出的。它的核心思想是<strong class="key-term">“信任域”</strong>。

为了防止步子太大扯到蛋,用旧策略 $\pi_{old}$ 收集的数据来评估改进方向。但这些数据<strong class="key-term">只在新策略 $\pi_{new}$ 和旧策略差异不大时才可信</strong>。 那么该如何量化“差异不大”？即我们后文会详细展开的 <strong class="key-term">KL 散度</strong>:$D_{KL}(\pi_{new} \| \pi_{old}) \le \delta$

总结来说叫信任域是因为<strong class="critical-term">你信任旧策略的数据在这个范围内仍然有效,超出这个范围,数据就不可信了</strong>。

#### 12.4.5.1 核心定义与符号

- <strong class="list-label">KL 散度</strong> : $D_{KL}(\pi_{\theta_{old}}(\cdot|s) || \pi_\theta(\cdot|s))$,衡量两个概率分布的距离。

  > $D_{KL}(P || Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx=∑​P(x)log\frac{P(x)}{q(x)}​$,具体知识比如和交叉熵的关系请参考之前的笔记

- <strong class="list-label">信任区域 (Trust Region)</strong>: 我们只相信在以旧策略为中心的一个小球内的更新。

#### 12.4.5.2 数学原理推导

TRPO 不再直接指定学习率$\alpha$,而是把优化问题转化为一个<strong class="key-term">带约束的优化问题</strong>。

我们希望最大化目标函数,但限制新旧策略之间的 KL 散度不超过 $\delta$:

$\max_{\theta}  \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}t \right]$

$\text{subject to} \quad \hat{\mathbb{E}}_t \left[ D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) || \pi_\theta(\cdot|s_t)) \right] \le \delta$

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么约束的是策略,而不是参数,我们不是要改进的就是参数吗？</p>

因为神经网络的参数和输出行为是<strong class="key-term">非线性关系</strong>,你约束了$\theta$,但最后输出的策略可能还是会超出我们接受的范围。<strong class="key-term">直接约束行为变化</strong>,而不是参数变化,那么无论你网络结构怎么变、参数怎么编码,只要行为(输出概率)变化在可接受范围内,我们就认为参数变化在可接受范围内。

</div>

由于这个约束优化难以直接求解。为了求解它,TRPO 做了以下近似:

1.  <strong class="list-label">一阶泰勒展开</strong>: 将目标函数进行一阶展开。

2.  <strong class="list-label">二阶泰勒展开</strong>: 将 KL 散度约束项进行二阶展开,近似为 $\frac{1}{2} (\theta - \theta_{old})^T H (\theta - \theta_{old})$,其中 $H$ 是 <strong class="key-term">费雪信息矩阵</strong>(具体过程见[对应补充节](/part-5/chapter-13#supp-trpo))。

#### 12.4.5.3 总结

TRPO的核心想法是”信任域”,也就是基于旧策略数据优化而来的新策略不可以与原来的旧策略差的太远。具体的优化过程基于黎曼几何空间的梯度下降,使用了标准梯度g乘上矫正的费雪信息矩阵$F^{-1}$也就是自然梯度方向。

#### 12.4.5.4 工业界落地的缺陷

- <strong class="list-label">计算灾难</strong>: F 是一个 $|\theta| \times |\theta|$ 的矩阵。如果网络有 100万个参数,计算 F 的逆矩阵 $F^{-1}$ 需要 $O(N^3)$ 的复杂度,太高了。

- <strong class="list-label">共轭梯度法</strong>: 虽然 TRPO 使用共轵梯度法(一种迭代的计算方法)来近似计算 $H^{-1}g$,避免了直接求逆,但实现起来依然非常复杂且慢。

## 12.5 PPO

从本质上讲,TRPO 通过在参数空间施加“<strong class="key-term">硬约束</strong>”(即 <strong class="key-term">KL 散度约束</strong>)来保证策略更新的<strong class="key-term">单调性</strong>,这是一个优雅的二阶优化问题。然而,在工程实践中,计算 Hessian 矩阵的逆(或使用共轭梯度法)代价极其昂贵。

<strong class="critical-term">PPO 的核心使命,就是用简化的方法,去逼近 TRPO 的二阶性能。</strong> 也就是把 TRPO 的“硬约束”转化为目标函数中的“<strong class="key-term">截断机制</strong>”。

### 12.5.1 核心定义与符号体系

- <strong class="list-label">$\pi_\theta(a|s)$</strong>:当前待优化的策略网络,参数为 $\theta$。

- <strong class="list-label">$\pi_{\theta_{\text{old}}}(a|s)$</strong>:在本次更新迭代前的旧策略,用于采样数据。

- <strong class="list-label">$\hat{A}_t$</strong>:在时间步 t 的优势函数估计值。通常使用 GAE 计算。

- <strong class="list-label">$r_t(\theta)$</strong>:<strong class="key-term">概率比率</strong>,这是 PPO 的核心变量: $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ <strong class="note-label">注意:</strong>当 $\theta = \theta_{\text{old}}$ 时,$r_t(\theta) = 1$。

- <strong class="list-label">$\epsilon$</strong>:超参数,截断范围,通常取值 0.1 或 0.2。

- <strong class="list-label">$J(\theta)$</strong>:保守策略迭代目标函数,即 TRPO 的无约束目标部分。

### 12.5.2 数学原理推导

#### 12.5.2.1 TRPO 简要回顾

TRPO 试图解决的问题是最大化以下目标函数,同时满足 KL 散度约束: $\max_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right]$

$\text{s.t.} \quad \hat{\mathbb{E}}_t \left[ D_{KL}(\pi{\theta_{\text{old}}}(\cdot|s_t) || \pi_\theta(\cdot|s_t)) \right] \le \delta$

直接对约束求解代价很高,PPO是直接对比率进行<strong class="key-term">截断clip</strong>。

#### 12.5.2.2 构建无约束代理目标

首先,我们定义无约束的基础目标函数: $J(\theta) = \hat{\mathbb{E}}_t [ r_t(\theta) \hat{A}_t ]$

如果我们直接最大化这个公式(不加约束),由于 $r_t(\theta)$ 可能变得非常大(新旧策略差异过大),会导致策略更新步长过大,从而发生策略坍塌。

#### 12.5.2.3 引入截断机制

PPO 的核心创新在于修改了 $J(\theta)$。它强制要求 $r_t(\theta)$ 保持在区间 $[1-\epsilon, 1+\epsilon]$ 附近。

我们需要定义一个<strong class="key-term">截断后的比率</strong>: $r_t^{\text{clip}}(\theta) = \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$

此时,PPO-Clip 的目标函数构造为:

<div class="key-formula">

$$
J^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]-\beta D_{KL}​(\pi_\theta​||\pi_{ref}​)
$$

</div>

其中第二项表示最新的模型不可以比参考模型偏的太远,$\beta$ 是调节的参数,现在比较流行使用<strong class="key-term">自适应的方式</strong>调节 $\beta$

$\beta_{t+1}=\begin{cases} \beta_t \cdot \alpha & D_{KL} > \text{target} \\ \beta_t / \alpha & D_{KL} < \text{target} \end{cases}$

这表示如果上一轮 $D_{KL} > \text{target}$ 即实际 KL 比预期大,说明,刚刚这轮更新,策略变化太大,因此增大$\beta$,加强这一项的惩罚。如果上一轮 $D_{KL} < \text{target}$ 即实际 KL 比预期小,说明策略变化不大,可以减少一下$\beta$,松弛一下惩罚。

#### 12.5.2.4 最小值情况拆开考虑

1.  <strong class="list-label">$\hat{A}_t > 0$(该动作优于平均,应增加概率)</strong>

    - 我们希望增加 $r_t(\theta)$(即 $\pi_\theta > \pi_{\theta_{\text{old}}}$)。

    - <strong class="list-label">目标函数变为:</strong>$\min(r_t(\theta)\hat{A}_t, (1+\epsilon)\hat{A}_t)$。

    - <strong class="list-label">数学含义</strong>:如果$r_t(\theta)$ 增长超过 $1+\epsilon$,目标函数就锁定在 $(1+\epsilon)\hat{A}_t$。这意味着<strong class="key-term">过大的更新不会带来额外的奖励,从而消除了让 $\theta$ 剧烈变化的梯度动力。</strong>

#### 12.5.2.5 最终完整损失函数

实际训练的损失函数和目标函数有些不同,我们还需要加入<strong class="key-term">价值函数损失 $L^{VF}$(用于更新 Critic)</strong>和<strong class="key-term">熵正则项 S(用于鼓励探索)</strong>:

$L_t^{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S\pi_\theta \right]$

其中 $c_1, c_2$ 是系数。

$L_t^{VF}(θ)=(V_θ(s_t)−V_t^{target})^2$

$V_t^{target}$ 通常来自经过GAE操作后的样本

#### 12.5.2.6 熵的定义

对离散策略 $\pi(a|s)$:

$\mathcal{H}(\pi(\cdot|s))=-\sum_a \pi(a|s)\log \pi(a|s)$

<strong class="key-term">熵这一项反映了不确定性,其实就表示模型的探索程度,防止策略过早收敛。</strong>

当策略变成确定性的时候,熵变为0,也就没有探索了。当策略变成均匀分布,也就是每种选择概率都一致时,熵最大,探索程度最高。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么PPO的损失函数和目标函数不一样呢？</p>

目标函数: $J^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]-\beta D_{KL}(\pi_\theta||\pi_{ref})$

这是经过我们理论上得到的最优结果,第一项是限制策略不比上一步偏差太多,第二项是利用KL散度来限制模型不要比参考模型差太多,实际上这个可以经过 <strong class="key-term">reward shaping</strong> 变成一项。

$J^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]-\beta \mathbb{E}[\log \frac{\pi_\theta}{\pi_{ref}}]$,去掉期望E,实际上就变为了r = $r_{clip} - \beta \log\frac{\pi_\theta}{\pi_{ref}}$成了一项。

而实际在工程上,为了更好训练神经网络,我们要加一些辅助的loss,一方面要考虑到Critic网络,因此补充上价值函数的损失；另一方面我们不希望策略尽早收敛,所以补充上了熵。

$L_t^{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S\pi_\theta \right]$

标准的PPO实现里是把目标函数的KL散度以及$\beta$自适应策略放到外面,如果KL偏差太大,直接<strong class="key-term">早停</strong>。但我查阅的一些工业界的代码里面,也有把KL项放到损失函数里面的

$Loss = -\hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta)\right] + c_1 L_t^{VF}(\theta) - c_2 S\pi_\theta +\beta D_{KL}$

</div>
