# iHMM (HDP‑HMM / Sticky‑HDP‑HMM) — MATLAB‑Parity Python Skeleton

本仓库是为 **先用 Python 调试、后逐行转译为 MATLAB** 的工作流而设计的 **一一对齐** 骨架：
- 目录/文件名/函数名 与 MATLAB 版本严格一致（每个文件一个顶层函数）。
- 数据结构使用 `dict`（相当于 MATLAB `struct`），便于无痛翻译。
- 采样器与发射族通过**函数集合**而不是类进行调度（等价于 MATLAB 的函数句柄）。

> ⚠️ 当前仓库为 **骨架**：大多函数仍是 stub（空实现）。你可以按本文档的接口契约逐步填充。

---

## 如何运行（建议步骤）

1. **克隆或解压**本项目；确保 Python ≥ 3.9，推荐创建虚拟环境：
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install numpy matplotlib
   ```

2. **实现最小闭环**（示例：弱极限 + 对角高斯）：
   - 填充以下文件（可参考 MATLAB 一侧实现或你已有代码）：
     - `emissions/gauss_diag_{init,loglik,suffstats,posterior,sample}.py`
     - `samplers/{sampler_da or sampler_beam}.py`
     - `samplers/ffbs_slice_sample_z.py`（若用 beam/slice）或 `samplers/ffbs_truncated.py`（若自行添加弱极限 FFBS）
     - `samplers/calc_loglik_current.py`
   - 运行示例：
     ```bash
     python -c "from ihmm.examples.ex_gauss1d_da import ex_gauss1d_da; ex_gauss1d_da()"
     ```

3. **在 Python 中调参/调试**直至收敛、指标正常；随后做**逐行语法转译**到 MATLAB：
   - `def f(...)` → `function ... end`
   - 导入 → `addpath` 或直接同名调用
   - 注意 0/1 基索引差异与 `axis`/`dim` 参数对应

---

## 采样器选择

- `sampler = 'da'`（Direct‑Assignment 折叠吉布斯）  
  - **优点**：共轭族下收敛性与数值稳定性好，易实现；
  - **缺点**：需维护/采样 CRF 桌数 `m_{kl}`，大计数时略慢。

- `sampler = 'beam'`（Beam/Slice，Van Gael 等）  
  - **优点**：真·无限状态，动态活跃集，复杂度接近与活跃状态数相关；
  - **缺点**：实现更复杂；需要 slice 变量与活跃集维护。

> 若只为尽快跑通：先用 **弱极限**近似（固定较大 K，Dirichlet 更新），等价于 `sampler='weaklimit'` 的实现思路。

---

## 发射族接口（Emission Family API）

每个发射族 `<fam>` 需在 `emissions/` 下提供 **五件套**（Python 文件名与 MATLAB `.m` 一致）：

- `<fam>_init(Y, prior)` → `theta`（单状态参数）  
- `<fam>_loglik(Y, theta_set)` → `LL`（形状 `K×T` 的对数似然）  
- `<fam>_suffstats(Y, z)` → `SS`（充分统计）  
- `<fam>_posterior(SS, prior)` → `post`（后验超参/条件）  
- `<fam>_sample(post)` → `theta`（从后验采样参数）  

### 典型族与先验示例
- `gauss_diag`（对角高斯，NIX 先验）：
  ```python
  prior = {
      'mu0': np.zeros(D),
      'kappa0': 1e-2*np.ones(D),
      'a0': 2*np.ones(D),   # IG shape
      'b0': 1*np.ones(D),   # IG rate
  }
  ```
- `poisson`（计数型，Gamma-Poisson）：
  ```python
  prior = { 'a0': 1.0, 'b0': 1.0 }  # rate ~ Gamma(a0,b0); lambda ~ Gamma posterior
  ```

**Y 的约定**：`list[np.ndarray]`，每条为形状 `(T, D)` 的 `float64`/`int64`。  
**theta_set**：长度 `K` 的参数 `dict` 列表。

---

## 统一数据结构（约定建议）

### `state`（训练中间态/最终模型）
- `beta` (`1×K`)：全局 GEM 权重（K 动态）  
- `Pi` (`K×K`)：当前活跃行/列的转移矩阵（或其充分统计）  
- `alpha, gamma, kappa`：超参  
- `theta`：长度 `K` 的参数结构体列表（每个状态的发射参数）  
- `z`：`list[np.ndarray]`，每条序列的状态路径 `(T,)`  
- `counts`：CRF 计数（`n_jk`）与表数（`m_jk`）  
- `active`：活跃状态索引/掩码（例如 `np.array([True, False, ...])` 或索引列表）

### `samples`（按 `opt.save_every` 保存的采样轨迹）
- `z, beta, alpha/gamma/kappa,（可选）Pi，theta，loglik，K_active`

### `opt`（关键项与常用超参）
- `sampler`: `'da' | 'beam' | 'weaklimit'`
- `n_iter, burnin, thin`
- `sticky: bool`, `kappa_init`（或固定 `kappa`）
- `emission.family = 'gauss_diag' | 'poisson' | ...` 与对应 `prior`
- 生命周期策略：`state_birth_thresh, prune_min_count`  
- 运行控制：`seed, verbose, save_every`

---

## 目录速览

```
ihmm/
  core/        训练入口与初始化
  samplers/    DA / Beam / 弱极限 与 FFBS / slice
  hdp/         CRF 计数、stick‑breaking、Dirichlet 抽样
  emissions/   发射族五件套（gauss_diag、poisson 等）
  utils/       logsumexp、采样、重标号等
  diagnostics/ 轨迹/热图/指标
  io/          数据加载与结果导出
  examples/    最小示例
  tests/       单测
```

---

## 复现性与数值约定

- 随机流统一：`utils/rng_manager.py`；MATLAB 侧用 `rng(seed,'twister')`。  
- 所有概率运算尽量在 log 域完成：`utils/logsumexp.py`, `utils/normalize_log.py`。  
- Inverse‑Gamma 采样：`1 / Gamma(shape=a, scale=1/b)`（**注意 rate/scale 参数化**）。  
- 概率下限：对 `beta, Pi`、似然矩阵做 `max(x, eps)` 以防止 `log(0)`。

---

## MATLAB 转译提示

- 函数一一对应；Python 导入改为 MATLAB 路径调用。  
- 轴/维度：`axis=0/1` ↔ MATLAB `dim=1/2`；保持张量形状 `(K, T)`、`(T, D)` 的一致性。  
- 索引：Python 0 基 → MATLAB 1 基，慎重处理 `range(T-1)` 等。

---

## 许可与引用

- 学术实现可参考：
  - Teh et al., 2006. *Hierarchical Dirichlet Processes.*
  - Fox et al., 2011. *A Sticky HDP‑HMM with Bayesian Structure Learning.*
  - Van Gael et al., 2008. *Beam Sampling for the Infinite HMM.*

