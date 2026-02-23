# CC-CRC Long-Tail (CIFAR100-LT)

## CVaR-over-Classes Conformal Risk Control — Detailed Idea & Implementation Plan (paper-ready)

> Goal: end-to-end implementation (no mocking) to generate **sufficient tables/figures/metrics** for writing a paper on **long-tailed / imbalanced classes** problems, with a new reliability target: **protect worst classes** through **CVaR over classes** within the **Conformal Risk Control (CRC)** framework.

---

# 1) Detailed Idea

## 1.1 Context & Gap

### Long-tail reliability is a "wrong target problem" if using rigid per-class coverage

In long-tailed classification (fraud/medical defect/rare errors), tail classes have very few calibration samples. If we require **per-class coverage** for *all classes* (Mondrian/class-conditional CP), then to satisfy the constraint for the rarest class, prediction sets often have to **inflate extremely** → losing efficiency.

### Practical Gap

* **Marginal CP/CRC**: small sets but tail classes are often under-covered.
* **Class-conditional CP**: guarantees class-wise coverage but pays the price of extremely large set sizes when the tail is heavy.

**Better objective**: protect the worst group of classes instead of all classes with a rigid threshold.

---

## 1.2 New Target: CVaR-over-classes Reliability

Let (K) be the number of classes. With a family of prediction sets (\Gamma_\lambda(x)) monotone in (\lambda) ((\lambda) increases → larger sets).

### Per-class miscoverage (risk per class)

[
m_y(\lambda) = \Pr{Y \notin \Gamma_\lambda(X)\mid Y=y}\in[0,1]
]
Risk vector per class: (m(\lambda)=(m_1(\lambda),...,m_K(\lambda))).

### CVaR-over-classes

We define the reliability objective:
[
\mathrm{CVaR}_\rho(m(\lambda)) \le \alpha
]

* (\rho\in(0,1]): tail level (e.g., (\rho=0.2) means "average of the worst 20% classes").
* (\alpha): target miscoverage.

**Interpretation**: instead of forcing all classes to satisfy ≤ α, we ensure the **average of the worst classes** (worst (\rho) fraction classes) does not exceed α. This is a "protect worst classes" statement that is highly reasonable for long-tail scenarios.

---

## 1.3 Algorithm Idea: CC-CRC (CVaR-over-classes CRC)

### Core Trick

CVaR has the form "worst (\rho)-classes average". We will:

1. Estimate an **upper bound** for miscoverage of each class (to avoid noise in the tail).
2. Select the worst (\rho)-fraction classes according to the upper bound.
3. On an independent split, check feasibility using a CRC-style bound on the weighted loss.

### Data Split to Avoid Double-Dipping

Split calibration into 2 independent parts:

* **C1**: estimate classwise risk + select worst classes.
* **C2**: test the constraint (conformal/CRC-style) to ensure distribution-free guarantees.

---

## 1.4 Loss Definition to Reduce to CRC

For a fixed (\lambda):

* Miscoverage indicator: (\ell_i(\lambda)=\mathbf{1}{y_i\notin\Gamma_\lambda(x_i)}).
* Class weights:
  [
  w(y)=\begin{cases}
  1/\rho & y \in \text{Worst}_\rho \
  0 & \text{otherwise}
  \end{cases}
  ]
* Define sample-level violation loss trên C2:
  [
  V_i(\lambda)= w(y_i),(\ell_i(\lambda)-\alpha)
  ]
  Khi (\lambda) tăng, (\ell_i(\lambda)) giảm (monotone), nên (V_i(\lambda)) cũng monotone.

**Feasibility check (CRC-style)**: chọn (\hat\lambda) nhỏ nhất sao cho upper confidence bound của mean(V) ≤ 0.

---

## 1.5 Why This is "Novel" & Paper-Worthy

1. **New reliability target** for long-tail: CVaR over classes (tail risk per class) instead of marginal or uniform classwise.
2. **Reduction to CRC**: transform the class-tail constraint into a sample-level monotone bounded loss to choose (\lambda) distribution-free via bounds.
3. **Efficiency**: avoid set size explosion due to extremely rare tail classes.
4. **Extensibility**: from CIFAR100-LT → ImageNet-LT/iNat, and from 1-prob score → APS/RAPS.

---

# 2) Implementation Plan (paper-ready, detailed, step-by-step control)

## 2.1 Implementation Principles

* **Conda-first**, no venv.
* Run using **CLI commands** (console scripts), no need to export PYTHONPATH.
* Each step is a small commit with clear **acceptance criteria**.
* Standardized artifacts for paper: metrics JSON + figures PNG + summary CSV + aggregated CSV.

---

## 2.2 CLI & Conda Packaging

### Environment

* `environment.yml` (python 3.11, pytorch/torchvision, numpy/scipy/sklearn/matplotlib/tqdm)

### Console Scripts

* `cccrc-train` : train backbone
* `cccrc-dump`  : dump probs/logits
* `cccrc-run`   : run method + eval + figures
* `cccrc-sweep` : sweep grid + summary
* (optional) `cccrc-data-inspect`, `cccrc-print-table`

### Install workflow

```bash
conda env create -f environment.yml
conda activate cccrc-longtail
pip install -e .
cccrc-train --help
```

---

## 2.3 Artifact Contract (for paper writing)

Each run saves to:
`artifacts/cifar100lt_IF{IF}_seed{seed}/`

### Required files

1. `splits.json` (train_fit/calib/C1/C2/test + class_counts_train + lt_indices)
2. `model.pt` + `train_report.json`
3. `probs_C1.npz`, `probs_C2.npz`, `probs_test.npz`
4. `method_{method}_...json` (tham số fit)
5. `metrics_{method}_...json` (full metrics)

### Required figures (PNG)

* `fig_miscoverage_rank_{method}_...png`
* `fig_tradeoff_{method}_IF{IF}_rho{rho}.png`
* `fig_bucket_bars_{method}_...png`
* `fig_cdf_miscoverage_{method}_...png`

### Summary tables

* `artifacts/summary.csv` (row-level)
* `artifacts/summary_agg.csv` (mean±std across seeds)
* optional `artifacts/table_main.tex`

---

# 3) Implementation Roadmap in Small Steps (Cursor-friendly)

> Each step below is an independent prompt for Cursor. Only move to the next step when the current one passes acceptance.

## M0 — Scaffold + CLI (2 steps)

### Step M0.1 — Repo scaffold + conda env

**Deliverables**: `environment.yml`, `pyproject.toml`, skeleton package, console scripts stubs.

**Acceptance**:

* `conda env create -f environment.yml`
* `pip install -e .`
* `cccrc-train --help` chạy được.

### Step M0.2 — CLI plumbing

Implement argparse/typer for the 4 commands, chỉ parsing args và print.

**Acceptance**:

* `cccrc-train --if 100 --seed 0 --epochs 1` in “Not implemented yet” nhưng không crash.

---

## M1 — Data CIFAR100-LT (3 steps)

### Step M1.1 — Artifacts I/O utils

Implement `infra/io/artifacts.py` (save/load json/npz/ckpt).

**Acceptance**:

* script nhỏ save/load npz và json OK.

### Step M1.2 — CIFAR100-LT generator + splits

Implement `infra/data/cifar_lt.py`:

* exponential downsample by IF
* stratified split LT→train_fit/calib
* stratified split calib→C1/C2
* save/load splits
* sanity checks (disjointness, sizes, min count)

**Acceptance**:

* `cccrc-data-inspect --if 100 --seed 0` tạo `splits.json` và print thống kê counts.

### Step M1.3 — Data inspect CLI

Add `cccrc-data-inspect` command.

**Acceptance**:

* chạy lệnh và tạo artifacts folder.

---

## M2 — Model + training (4 steps)

### Step M2.1 — ResNet32 CIFAR implementation

Implement `infra/models/resnet_cifar.py` + sanity forward.

**Acceptance**:

* forward (2,3,32,32) -> (2,100)

### Step M2.2 — Training loop minimal

Implement `cccrc-train`:

* load/generate splits
* transforms: random crop + flip + normalize
* train on train_fit
* eval on test
* save best ckpt `model.pt`

**Acceptance**:

* `cccrc-train --if 10 --seed 0 --epochs 2` tạo `model.pt`

### Step M2.3 — Deterministic & report

Add `set_seed()`, save `train_report.json` (best_acc, epoch, IF, seed).

**Acceptance**:

* 2 runs with the same seed create identical splits.

### Step M2.4 — Quick accuracy verify

Add `cccrc-verify` hoặc logic in train để verify ckpt.

---

## M3 — Dump probs (2 steps)

### Step M3.1 — Implement `cccrc-dump`

* load ckpt + splits
* infer probs for C1/C2/test
* save npz
* sanity: probs sum ~ 1, shapes.

**Acceptance**:

* `cccrc-dump --if 10 --seed 0` tạo 3 npz.

### Step M3.2 — Parity check

Compute top1 acc from probs_test and compare with train_report.

---

## M4 — Domain primitives + Baselines (5 steps)

### Step M4.1 — Core domain functions

Implement:

* `domain/scores.py` (1-prob)
* `domain/sets.py`
* `domain/risk.py` (CVaR + worst-classes)

### Step M4.2 — Evaluation metrics

Implement `evaluate()`:

* per-class miscoverage, CVaR_rho
* mean/median set size
* head/mid/tail bucket metrics

### Step M4.3 — Split-CP

Implement fit/predict.

### Step M4.4 — Mondrian CP

Implement per-class thresholds.

### Step M4.5 — Bucket CP

Implement buckets by train frequency + thresholds.

**Acceptance**:

* `cccrc-run --method split_cp --alpha 0.1` tạo metrics JSON.

---

## M5 — CC-CRC method + figures (6 steps)

### Step M5.1 — Bounds (Hoeffding UCB)

Implement `domain/bounds.py`.

### Step M5.2 — CC-CRC fit

Implement `domain/methods/cc_crc.py`:

* lambda grid
* C1 UCB per class
* worst classes
* C2 weighted loss V
* choose first lambda where UCB(mean(V)) ≤ 0

### Step M5.3 — Runner CLI

Implement `cccrc-run`:

* load probs
* fit method
* predict set on test
* compute metrics
* save method json + metrics json

### Step M5.4 — Figures: miscoverage rank

Implement `fig_miscoverage_rank`.

### Step M5.5 — Figures: CDF + bucket bars

Implement 2 plots:

* CDF of per-class miscoverage
* head/mid/tail bars for miscoverage and set size

### Step M5.6 — Tradeoff curve

Implement `cccrc-tradeoff` (or inside `cccrc-run --tradeoff`):

* sweep alpha ∈ {0.05,0.1,0.2}
* plot CVaR_rho vs mean set size

**Acceptance**:

* `cccrc-run --method cc_crc --alpha 0.1 --rho 0.2` tạo metrics + ≥3 figures.

---

## M6 — Sweep + paper tables (3 steps)

### Step M6.1 — Sweep driver

Implement `cccrc-sweep`:

* for each IF/seed ensure model and probs exist
* run each method and config
* write row-level `summary.csv`

### Step M6.2 — Aggregate table

Implement `cccrc-print-table`:

* groupby IF/method/alpha/rho
* output mean±std as `summary_agg.csv`

### Step M6.3 — Latex table export (optional)

Output `table_main.tex`.

**Acceptance**:

* `cccrc-sweep --ifs 10 50 100 --seeds 0 1 2 --alphas 0.05 0.1 0.2 --rhos 0.1 0.2 --methods split_cp mondrian_cp bucket_cp cc_crc`
* tạo `summary.csv` + `summary_agg.csv` + figures.

---

# 4) Experiment Design on CIFAR100-LT (sufficient for paper)

## 4.1 Datasets

* CIFAR100-LT với IF ∈ {10, 50, 100}
* Seeds ∈ {0,1,2}

## 4.2 Methods

* Split CP (marginal)
* Mondrian CP (class-conditional)
* Bucket CP (head/mid/tail)
* CC-CRC (alpha, rho)

## 4.3 Primary Metrics

* **CVaR_rho(per-class miscoverage)** (primary)
* Mean set size (efficiency)
* Marginal miscoverage (sanity)
* Head/mid/tail bucket miscoverage + set size

## 4.4 Primary Figures

1. Miscoverage vs frequency rank (highlight worst rho)
2. Tradeoff curve: CVaR_rho vs mean set size
3. Bucket bars (head/mid/tail)
4. CDF of per-class miscoverage

## 4.5 Table (paper)

Table main (mean±std over seeds) per IF:

* CVaR_rho
* Mean set size
* Marginal miscoverage

---

# 5) Important Notes (for "correct reliability" results)

1. **No double-dip**: CC-CRC must use C1 to select worst classes, C2 to check.
2. **Monotone family**: threshold-based (\Gamma_\lambda) ensures monotonicity.
3. **Tail class lacking calibration**: handle with conservative defaults (UCB=1 or q=1).
4. **Reproducibility**: save splits.json + seed + IF + transforms.

---

# 6) Extensions After CIFAR100-LT

* Score: APS/RAPS to reduce set size.
* Bound: Clopper–Pearson UCB for tighter bounds.
* Dataset: ImageNet-LT / iNaturalist.
* Theory: formalize CVaR-over-classes control + lower bound showing classwise uniform is too costly in long-tail.

---

# 7) "Done for paper writing" Checklist

You are DONE when:

* [ ] `cccrc-sweep` generates `summary_agg.csv` for IF=10/50/100, 3 seeds.
* [ ] Have all 4 figures for each IF and method.
* [ ] CC-CRC shows: CVaR_rho reduced compared to Split-CP, set size smaller than Mondrian.
* [ ] Ablation: rho (0.1/0.2) and alpha (0.05/0.1/0.2).

---

## Appendix: Prompt Template for Cursor (each step)

**Task**: (1 sentence)
**Files**: (create/modify)
**Constraints**: (conda+CLI, no mock, deterministic)
**Acceptance**: (commands to run)
**Expected artifacts**: (files/figures to create)
