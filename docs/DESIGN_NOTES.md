# DESIGN_NOTES — HDP‑HMM / Sticky‑HDP‑HMM

本文件记录模型变量、依赖关系、计数定义与实现要点，便于在 Python 与 MATLAB 之间保持一致。

---

## 1. 生成模型（带 sticky 自环偏置）

- 全局权重：`beta ~ GEM(gamma)`
- 行转移：对每个状态 `k`
  ```
  pi_k ~ DP(alpha + kappa * delta_k, beta)    # sticky: 自环项
  ```
- 初始状态与演化：
  ```
  z_1 ~ Cat(beta)
  z_t | z_{t-1} ~ Cat(pi_{z_{t-1}})
  y_t | z_t=k ~ F(theta_k),   theta_k ~ H
  ```

**弱极限近似（K 截断）**：
```
beta ~ Dir( gamma/K, ..., gamma/K )
pi_k ~ Dir( alpha*beta + kappa*e_k )
```
观测与转移计数纳入后验：
```
pi_k | counts ~ Dir( alpha*beta + kappa*e_k + N_{k*} )
beta  | tables ~ Dir( gamma/K + m_{·1}, ..., gamma/K + m_{·K} )
```

---

## 2. 计数定义（CRF）

- `N_{kl}`：从状态 `k` 转到 `l` 的次数（跨所有序列求和）。
- `n_{jk}`：序列 `j` 中到达/占据状态 `k` 的计数（可按需求定义）。
- **桌数 `m_{kl}`**：Chinese Restaurant Franchise 的“桌”计数，用于更新 `beta`：
  ```
  m_{kl} = sum_{i=1}^{N_{kl}} Bernoulli(  alpha * beta_l / (alpha * beta_l + i - 1)  )
  ```
  实现上可逐 `i` 抽样，或使用近似/向量化（N 较大时需注意性能）。

---

## 3. 推断算法概览

### 3.1 Direct‑Assignment（折叠吉布斯）
1. 给定 `beta, Pi, theta`，用 FFBS 对每条序列采样 `z`；统计 `N_{kl}`、起始计数等。
2. 采样 **桌数** `m_{kl}`，更新 `beta ~ Dir(gamma/K + m_{·l})`。
3. 更新每行转移：`pi_k ~ Dir(alpha*beta + kappa*e_k + N_{k*})`。
4. 按发射族共轭/条件更新 `theta_k`（`suffstats → posterior → sample`）。
5. （可选）更新超参 `alpha, gamma, kappa`（辅助变量或 slice）。

### 3.2 Beam/Slice（真·无限）
1. 为每个 `t` 采样切片变量 `u_t ∈ (0, π_{z_{t-1}, z_t})`；
2. 构造活跃集 `A_t = {k | π_{z_{t-1},k} > u_t}`；在 `A_t` 上做 FFBS；
3. 按活跃集统计计数并更新 `beta, pi_k, theta`；
4. 动态增删状态（birth/death/prune）。

---

## 4. 对数似然与前向‑后向（数值稳定）

- 观测对数似然：`LL ∈ R^{K×T}`，第 `k,t` 元为 `log p(y_t | z_t=k)`。
- 前向递推：
  ```
  alpha[:,1] = log beta + LL[:,1]
  alpha[:,t] = LL[:,t] + logsumexp( logA' + alpha[:,t-1]', axis=1 )
  log p(y_{1:T}) = logsumexp( alpha[:,T] )
  ```
- 采样回溯：使用条件 `p(z_t=k | z_{t+1}=l) ∝ exp(alpha[k,t]) * A[k,l]`。

---

## 5. 数值要点

- 对 `beta, Pi, LL` 统一施加最小值：`x = max(x, eps)`，避免 `log(0)`。
- `logsumexp`/`normalize_log` 自己实现，避免精度差异。
- Inverse‑Gamma 参数：本实现约定 `rate`（而非 `scale`）。Python: `1 / Gamma(a, scale=1/b)`；MATLAB: `1./gamrnd(a, 1./b)`。
- 对大 `N_{kl}` 的桌数采样可向量化或采用近似；若使用弱极限，可绕过 `m_{kl}` 而直接 Dirichlet 更新（但缺少 CRF 精确性）。

---

## 6. 统一数据结构（再次汇总）

### state
- `beta (1×K)`, `Pi (K×K)`, `alpha, gamma, kappa`, `theta (1×K structs)`
- `z`（list of `(T,)`），`counts`（`n_{jk}`/`N_{kl}` 与 `m_{kl}`），`active`（索引或布尔掩码）

### samples（按 `opt.save_every` 存）
- `z, beta, alpha/gamma/kappa,（可选）Pi，theta，loglik，K_active`

### opt（关键项）
- `sampler = 'da' | 'beam' | 'weaklimit'`
- `n_iter, burnin, thin`
- `sticky` 与 `kappa_init`（或固定 `kappa`）
- `emission.family` 与 `prior`
- 生命周期：`state_birth_thresh, prune_min_count`
- 运行：`seed, verbose, save_every`

---

## 7. 依赖关系（模块视角）

- `core/ihmm_fit` 调用 `samplers/*` 与 `emissions/*`，并读写 `state/samples`。
- `samplers/*` 依赖：`utils/logsumexp`, `utils/sample_discrete`, `hdp/dirichlet_draw`, `hdp/crf_*` 等。
- `emissions/*` 由 `samplers` 驱动：`suffstats → posterior → sample`；`loglik` 在 FF 及预测中广泛使用。
- `diagnostics/*` 只读 `samples/state` 生成图表与指标；`io/*` 负责读写。

---

## 8. MATLAB ↔ Python 一致性备忘

- 形状：`(K, T)` 与 `(T, D)` 固定约定；不要依赖 Python 广播“魔法”。
- 索引：Python 0‑base ↔ MATLAB 1‑base；尤其在 `for t in range(T-1)` 上谨慎翻译。
- 随机流：统一入口，算法不同不保证逐样本一致，但宏观统计应一致。

