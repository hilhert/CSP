### Single CSP Block

Each block has four stages:

    z_t ──► Rotate ──► Recur ──┐
         │                     │
         └───── Skip ──────────┼──► (+) ──► Norm ──► h_t

- **Rotate**: `z~_t = e^(i*theta_t) ⊙ z_t`
- **Recur**: `h_t = alpha_t * h_{t-1} + gamma_t * z~_t`
- **Skip**: `h~_t = h_t + z_t`
- **Norm**: `h_t^(l) = h~_t / |h~_t|`

# Complex State Propagator (CSP)

**State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking**

[![arXiv](https://img.shields.io/badge/arXiv-2608.03425-b31b1b.svg)](https://arxiv.org/abs/2608.03425)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## TL;DR

We propose **CSP (Complex State Propagator)**, a minimalistic recurrent architecture that **only propagates hidden states** across layers, without output projections at intermediate steps. The state is complex-valued and updated via learned rotations.

On the training length (T = 16), CSP reaches **100% accuracy** on Parity Check, Mod-3 Counting, and Parenthesis Matching. Length generalization is evaluated separately: models are trained at T = 16 and tested up to T = 64 (4× the training length). See the Results section below for the full comparison against five baselines.

**Key insight:** In standard Mamba, `h -> y = C h -> next_h = B(y) = B C h`. The two projections can be fused. Why not just propagate `h` directly?

If you find this work useful, please cite it in your paper:

    @article{li2026state,
      title={State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking},
      author={Li, Xiaohe and Lu, Yang},
      journal={arXiv preprint arXiv:2608.03425},
      year={2026}
    }

---

> **Development Status**
>
> The CSP family is **under active development**. This repository tracks the latest experimental results and may not be fully consistent with the version described in our arXiv paper.
>
> Two variants are currently maintained:
>
> - **CSP-Vanilla** — the iterative reference implementation, with state rotation applied through a sequential recurrence.
> - **CSP-Fast** — a parallel formulation using accumulated-phase rotation sensing.
>
> The rotation projection was recently modified from `tanh(·)·π` to `atan2(·)`, which removes an artificial magnitude bound on the rotation angle. All results below reflect this change.

---

## Dataset Note

The Parenthesis dataset construction uses the **Dyck-1 style** format:

- Standard valid-prefix labeling
- Tokens remapped to embedding indices (left = 1, right = 2, pad = 0)
- Labels: `1` if the prefix is a valid Dyck-1 prefix, `0` otherwise, `-100` for padding

Older checkpoints may use a different construction and are not directly comparable.

---

## Architecture

CSP is a minimal recurrent architecture that **only propagates hidden states** across layers, without output projections at intermediate steps.

### Overall Flow

    Input x (T steps)
        │
        ▼
      Embedding
        │
        ▼
      CSP Block 1 ──► h^(1)
        │
        ▼
      CSP Block 2 ──► h^(2)
        │
        ▼
        ...
        │
        ▼
      CSP Block L ──► h^(L)
        │
        ▼
      Phase Decoder (atan2 -> [cos, sin])
        │
        ▼
      Output y_hat

### Key Principles

1. **State-Only Propagation** – No output projections between layers.
2. **Complex Rotation** – Rotation angle extracted as `atan2(W_θ z_t)`, without magnitude bounding.
3. **Phase Decoding** – Final prediction from `atan2(Im(h), Re(h))`.
4. **Block-Level SiLU** – Nonlinearity only at block boundaries.

### CSP-Fast: Accumulated-Phase Variant

CSP-Fast computes the per-step rotation angle directly from the **accumulated phase** of the input sequence:

    theta_all = cumsum( atan2( theta_proj( x_t ) ) )

The `atan2` projection maps the input to a 2D point whose angle is the phase increment. Since the rotation angle is a direct function of the running cumulative sum, it can be evaluated **in parallel** across all timesteps with a single `cumsum` operation.

Key properties:

- **Accumulated phase.** The rotation angle at each step is derived from the running sum of per-step phase increments.
- **Complex-eigenvalue structure.** The state transition matrix `A = a * e^(i*theta)` remains complex-valued.
- **No magnitude bounding.** The phase is extracted from the projection via `atan2`, so the magnitude of the projection is irrelevant. This removes the `tanh(·)·π` constraint present in earlier versions.
- **Parallel computation.** The angle is computed in one pass over the sequence.

---

## Results

### Training (fixed length, T = 16)

Accuracy at the training length only. Length generalization is reported separately below.

| Task | Accuracy | F1 | Epochs to 100% |
|------|----------|-----|----------------|
| Parity Check | 100% | 1.0 | ~10 |
| Mod-3 Counting | 100% | 1.0 | ~10 |
| Parenthesis Matching | 100% | 1.0 | ~10 |

### Length Generalization

Trained at T = 16, evaluated up to T = 64 (4× training length). All values are **mean ± std across five seeds (42–46)**.

#### Parity

| Model | Params | L=16 | L=32 | L=64 |
|-------|--------|------|------|------|
| Vanilla RNN | 2,754 | 1.000 | 1.000 | **1.000** |
| Complex RNN | 10,242 | 1.000 | 0.999 | 0.994 ± 0.011 |
| CSP-Vanilla | 4,686 | 1.000 | 0.922 | 0.716 ± 0.011 |
| CSP-Fast | 4,590 | 0.962 | 0.790 | 0.645 ± 0.032 |
| Mamba (neg. eigen) | 18,693 | 0.967 | 0.776 | 0.638 ± 0.018 |
| Transformer | 41,186 | 0.976 | 0.727 | 0.600 ± 0.014 |

#### Mod-3 Counting

| Model | Params | L=16 | L=32 | L=64 |
|-------|--------|------|------|------|
| Vanilla RNN | 2,803 | 1.000 | 1.000 | **1.000** |
| Complex RNN | 10,323 | 1.000 | 0.996 | 0.983 ± 0.038 |
| CSP-Vanilla | 4,893 | 1.000 | 0.980 | 0.921 ± 0.145 |
| CSP-Fast | 4,893 | 1.000 | 0.974 | 0.876 ± 0.171 |
| Mamba (neg. eigen) | 18,742 | 1.000 | 0.803 | 0.565 ± 0.028 |
| Transformer | 41,235 | 0.999 | 0.646 | 0.475 ± 0.011 |

#### Dyck-1 (Parenthesis)

| Model | Params | L=16 | L=32 | L=64 |
|-------|--------|------|------|------|
| Transformer | 41,202 | 1.000 | 0.995 | **0.983 ± 0.003** |
| Vanilla RNN | 2,770 | 1.000 | 0.999 | 0.990 ± 0.004 |
| Complex RNN | 10,258 | 1.000 | 1.000 | 0.984 ± 0.020 |
| CSP-Vanilla | 4,795 | 0.999 | 0.972 | 0.857 ± 0.039 |
| CSP-Fast | 4,795 | 0.993 | 0.966 | 0.919 ± 0.040 |
| Mamba (neg. eigen) | 18,709 | 1.000 | 0.981 | 0.900 ± 0.063 |

### Observations

**Three-tier behavior.** Across all three tasks, models split into two groups. *Exact-transition* models (Vanilla RNN, Complex RNN) reach L=64 accuracy ≥ 0.98 on parity and mod-3, and ≥ 0.98 on Dyck-1. *Soft-contraction* models (CSP, Mamba, Transformer) decay with sequence length.

**Mod-3 is the sharpest discriminator.** With the atan2 projection, CSP-Vanilla and CSP-Fast reach 0.921 and 0.876 at L=64 on mod-3, well above Mamba (0.565) and the Transformer (0.475). Note the high variance across seeds: CSP-Vanilla std = 0.145, CSP-Fast std = 0.171. Some seeds reach perfect 1.000 length generalization.

**Dyck-1 favors the Transformer.** The Transformer achieves 0.983 at L=64, the highest of any model. CSP remains competitive, with CSP-Fast at 0.919 and CSP-Vanilla at 0.857.

**Parity is mid-tier for CSP.** On parity, CSP-Vanilla (0.716) and CSP-Fast (0.645) outperform Mamba (0.638) and the Transformer (0.600), but do not reach the exact-transition tier.

### Grokking Phenomenon

We observe clear **grokking** on all three tasks: performance stays near chance for many epochs, then jumps sharply.

**Parity**

<div align="center">
  <img src="./experiments/parity/CSP-Fast/figure/grokking_analysis.png" width="400"/>
  <img src="./experiments/parity/CSP-Fast/figure/training_curves.png" width="400"/>
</div>

**Mod-3 Counting**

<div align="center">
  <img src="./experiments/mod3/CSP-Fast/figure/grokking_analysis.png" width="400"/>
  <img src="./experiments/mod3/CSP-Fast/figure/training_curves.png" width="400"/>
</div>

**Dyck-1**

<div align="center">
  <img src="./experiments/Dyck-1/CSP-Fast/figure/grokking_analysis.png" width="400"/>
  <img src="./experiments/Dyck-1/CSP-Fast/figure/training_curves.png" width="400"/>
</div>

**Length generalization curves (all seeds)**

**Seed 42**

<div align="center">
  <img src="./experiments/parity/results/parity_length_acc_seed_42.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/mod3/results/mod3_length_acc_seed_42.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/Dyck-1/results/Dyck-1_length_acc_seed_42.png" width="600"/>
</div>

**Seed 43**

<div align="center">
  <img src="./experiments/parity/results/parity_length_acc_seed_43.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/mod3/results/mod3_length_acc_seed_43.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/Dyck-1/results/Dyck-1_length_acc_seed_43.png" width="600"/>
</div>

**Seed 44**

<div align="center">
  <img src="./experiments/parity/results/parity_length_acc_seed_44.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/mod3/results/mod3_length_acc_seed_44.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/Dyck-1/results/Dyck-1_length_acc_seed_44.png" width="600"/>
</div>

**Seed 45**

<div align="center">
  <img src="./experiments/parity/results/parity_length_acc_seed_45.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/mod3/results/mod3_length_acc_seed_45.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/Dyck-1/results/Dyck-1_length_acc_seed_45.png" width="600"/>
</div>

**Seed 46**

<div align="center">
  <img src="./experiments/parity/results/parity_length_acc_seed_46.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/mod3/results/mod3_length_acc_seed_46.png" width="600"/>
</div>

<div align="center">
  <img src="./experiments/Dyck-1/results/Dyck-1_length_acc_seed_46.png" width="600"/>
</div>

---

## Quick Start

    # Clone
    git clone https://github.com/hilhert/CSP.git
    cd CSP

    # Install
    pip install -r requirements.txt
    pip install -e .

    # Run experiments
    python experiments/parity.py
    python experiments/mod3.py
    python experiments/parenthesis.py

    # Jupyter demo
    jupyter notebook notebooks/demo.ipynb

## Requirements

- torch>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- tqdm>=4.65.0
- scikit-learn>=1.3.0
- safetensors>=0.8.0

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Xiaohe Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
