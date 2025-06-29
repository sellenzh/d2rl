# 策略梯度定理：

## 定义

当轨迹中某个 ‘状态-动作对’ $(s,a)$ 是令人满意的，即 $Q(s,a) > 0$，那就增加 $\pi_\theta(s,a)$ 的概率。



## 证明

策略梯度定理 有两种形式：
$$
\nabla_\theta J(\theta) = E_{\substack{s_t \sim \mathrm{Pr}(s_0 \to s_t, t, \pi) \\ a_t \sim \pi(a_t|s_t)}} \left[ \sum _{t=0} ^{\infty} \gamma ^t Q_\pi(s_t,a_t) \nabla \ln \pi(a_t | s_t)  \right]
$$

$$
\nabla_\theta J(\theta) \propto E_{\substack{s \sim D^\pi \\ a \sim \pi(a|s)}} \left[ Q_{\pi_\theta}(s,a) \nabla_\theta \ln \pi_\theta(a|s) \right]
$$

首先，定义强化学习的优化目标：$J(\theta) \doteq V_{\pi_\theta}(s_0)$，其中，$v_{\pi_\theta}$ 是 $\pi_\theta$ 的真实价值函数。目标函数求梯度有：
$$
\begin{align*}
\nabla_\theta J(\theta) &= \nabla_\theta V_{\pi_\theta}(s_0) = \nabla\left[\sum_{a_0} \pi(a_0|s_0) Q_\pi(s_0, a_0) \right] \\
&= \sum_{a_0} \left[\nabla \pi(a_0 | s_0) Q_\pi(s_0, a_0) + \pi(a_0|s_0) \nabla Q_\pi(s_0, a_0) \right] \\
&= \sum_{a_0} \left[\nabla \pi(a_0 | s_0) Q_\pi(s_0, a_0) + \pi(a_0|s_0) \nabla \sum_{s_1, r_1}p(s_1, r_1|s_0, a_0)(r_1 + \gamma V(s_1)) \right] \\
&= \sum_{a_0} \nabla \pi(a_0|s_0)Q_\pi(s_0, a_0) + \sum_{a_0}\pi(a_0|s_0) \sum_{s_1}p(s_1|s_0,a_0) \cdot \gamma \nabla V(s_1) \\
&= \sum_{a_0} \nabla \pi(a_0 | s_0) Q_\pi(s_0, a_0) + \sum_{a_0} \pi(a_0 | s_0) \sum_{s_1} p(s_1|s_0,a_0) \cdot \gamma \sum_{a_1} \left[ \nabla \pi(a_1|s_1) Q_\pi(s_1,a_1) + \pi(a_1|s_1) \sum_{s_2} p(s_2|s_1,a_1) \gamma \nabla V(s_2) \right] \\
&= \sum_{a_0} \nabla \pi(a_0 |s_0) Q_\pi(s_0,a_0) + \sum_{a_0} \pi(a_0|s_0)\sum_{s_1}p(s_1|s_0,a_0) \cdot \gamma \sum_{a_1} \nabla \pi(a_1|s_1)Q_\pi(s_1,a_1) + \dots \\
&= \sum_{s_0} \mathrm{Pr}(s_0 \to s_0, 0, \pi) \sum_{a_0} \nabla \pi(a_0 |s_0) \gamma^0 Q_\pi(s_0, a_0) + \sum_{s_1} \mathrm{Pr}(s_0 \to s_1, 1, \pi) \sum_{a_1} \nabla \pi(a_1|s_1) \gamma^1 Q_\pi(s_1, a_1) + \dots \\
&= \sum_{s_0} \mathrm{Pr}(s_0 \to s_0, 0, \pi) \sum_{a_0} \nabla \pi(a_0 |s_0) \left[\gamma^0 Q_\pi(s_0, a_0) \nabla \ln \pi(a_0|s_0)\right] + \sum_{s_1} \mathrm{Pr}(s_0 \to s_1, 1, \pi) \sum_{a_1} \nabla \pi(a_1|s_1) \left[ \gamma^1 Q_\pi(s_1, a_1) \nabla \ln \pi(a_1|s_1)\right]+ \dots \\
&= \sum_{t=0}^{\infty} \sum_{s_t} \mathrm{Pr}(s_0 \to s_t, t, \pi) \sum_{a_y} \pi(a_t|s_t)\left[\gamma^t Q_\pi(s_t,a_t) \nabla \ln \pi(a_t|s_t) \right]
\end{align*}
$$
其中，$p(s_1, r_1 | s_0, a_0)$ 表示环境转移概率， $\mathrm{Pr}(s_0\to s_t, t, \pi)$ 表示从状态 $s_0$ 出发，在策略 $\pi$ 的作用下，经过 $t$ 步到达状态 $s_t$ 的概率。如：
$$
\begin{align*}
\mathrm{Pr} (s_0 \to s_0, 0, \pi) &= 1 \\
\mathrm{Pr} (s_0 \to s_1, 1, \pi) &= \sum_{a_0} \pi(a_0|s_0) p(s_1|s_0,a_0) \\
\cdots
\end{align*}
$$
此外，还使用了 $\nabla \pi(a|s) = \pi(a|s) \nabla \ln \pi(a|s)$ 的恒等变换。基于此，我们得到 策略梯度定理 的基本形式：
$$
\begin{align*}
\nabla_\theta J(\theta) = \sum_{t=0} ^{\infty}\sum_{s_t} \mathrm{Pr}(s_0 \to s_t, t, \pi)\sum_{a_t} \pi(a_t|s_t) \left[\gamma^t Q_\pi(s_t, a_t) \nabla \ln \pi(a_t|s_t) \right]
\end{align*}
$$




