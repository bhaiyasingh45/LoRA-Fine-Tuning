# LoRA-Fine-Tuning

---

## Why does LoRA even exist? The problem first.

Imagine you want to fine-tune **Llama 3 (8B parameters)**. Full fine-tuning means updating all 8 billion weights. That needs ~120GB of GPU VRAM just for the gradients and optimizer states — completely out of reach for most teams.

LoRA asks a brilliant question: **do we really need to update all those parameters?**

The answer, backed by research, is **no** — because when models adapt to a new task, the *actual change* in weight matrices tends to live in a **very low-rank space**.

---

<img width="736" height="375" alt="image" src="https://github.com/user-attachments/assets/595c4653-4e15-4a56-a398-e2fc0f9fa097" />

---

## But first — why does fine-tuning eat so much memory? (an 8B walkthrough)

Before going into the LoRA math, it's worth understanding *why* full fine-tuning is so expensive. When you train an 8B model, the GPU doesn't just hold the weights — it has to keep **five separate buckets of memory** alive at the same time:

| Memory bucket | What it actually stores | Size for 8B in FP32 |
|---|---|---|
| **Model weights (W)** | The 8 billion parameters you're learning | 8B × 4 bytes = **32 GB** |
| **Gradients (∂L/∂W)** | One gradient value per parameter, computed in the backward pass | 8B × 4 bytes = **32 GB** |
| **Optimizer states** | AdamW keeps **two** FP32 momentum estimates per param: `m` (1st moment) and `v` (2nd moment) | 8B × 8 bytes = **64 GB** |
| **Activations** | Intermediate tensors from the forward pass, cached for use in backward | ~10–30 GB (depends on batch size & sequence length) |
| **Working buffers** | Communication ops, gradient checkpoints, fragmentation, CUDA overhead | ~2–5 GB |
| | **Grand total** | **~140–160 GB** |

That gives the famous **"AdamW 16× rule"**: full FP32 fine-tuning needs roughly **16 bytes per parameter** (4 weights + 4 gradients + 8 optimizer) — *before* you even account for activations.

### What about mixed precision (BF16 / FP16)?

You might think switching to BF16 halves the memory. It doesn't, because mixed-precision training still keeps an **FP32 master copy** of the weights for numerical stability:

- FP32 master weights → **4 bytes/param**
- BF16 working weights (used in forward/backward) → **2 bytes/param**
- BF16 (or FP32) gradients → **2–4 bytes/param**
- FP32 AdamW states (`m`, `v`) → **8 bytes/param**
- **Total: ~18 bytes/param + activations**

So mixed precision saves *some* memory, but the **optimizer states are the dominant cost**, not the weights themselves. That's the core reason LoRA is such a big deal: by training only ~0.1–1% of the parameters, the gradient + optimizer footprint collapses by 100×–1000×, even though the frozen base model is still sitting in VRAM.

---

## Precision formats — the bytes-per-parameter cheat sheet

| Format | Bits | Bytes/param | Dynamic range | Precision | Typical use |
|---|---|---|---|---|---|
| **FP32** | 32 | 4.0 | ±3.4 × 10³⁸ | ~7 decimal digits | Legacy training, FP32 master weights in mixed precision |
| **FP16** | 16 | 2.0 | ±65,504 (overflows easily) | ~3 decimal digits | Older mixed-precision (V100, T4); needs loss scaling |
| **BF16** | 16 | 2.0 | Same as FP32 | ~2–3 decimal digits | **Default for modern LLM training** (A100, H100, B200) |
| **FP8 (E4M3 / E5M2)** | 8 | 1.0 | Limited | Very low | Blackwell-era training, inference |
| **INT8** | 8 | 1.0 | Integer | Quantized | Inference, weight-only quantization |
| **INT4 / NF4** | 4 | 0.5 | Integer | Heavily quantized | **QLoRA base model storage**, low-RAM inference |

**Two things to remember:**

1. **Memory = parameter count × bytes-per-parameter.** That's the entire formula for weights.
2. **BF16 won the format war** because it has the *range* of FP32 (8 exponent bits, so gradients don't overflow) with the *footprint* of FP16 (2 bytes). FP16's tiny exponent (5 bits, max value ~65,504) causes gradient overflow during training — BF16 fixes that.

---

## VRAM requirements across model sizes & precisions

> All numbers assume **batch size 1**, **sequence length ≤512**, **AdamW optimizer**, **gradient checkpointing on**. Longer sequences scale activation memory roughly linearly. Using `paged_adamw_8bit` (bitsandbytes) cuts optimizer state by ~75%. Unsloth and FlashAttention-2 cut activation memory further.

### A. Inference / weights-only (no training)

Just `params × bytes/param`, then add ~20–30% for KV cache and CUDA overhead.

| Model | FP32 (4 B/p) | FP16 / BF16 (2 B/p) | INT8 (1 B/p) | INT4 / NF4 (0.5 B/p) |
|---|---|---|---|---|
| **1.5B** (Qwen2.5-1.5B) | 6 GB | 3 GB | 1.5 GB | 0.75 GB |
| **3B** (Llama-3.2-3B) | 12 GB | 6 GB | 3 GB | 1.5 GB |
| **7B** (Llama-3-7B, Qwen-7B) | 28 GB | 14 GB | 7 GB | 3.5 GB |
| **8B** (Llama-3-8B) | 32 GB | 16 GB | 8 GB | 4 GB |
| **13B** | 52 GB | 26 GB | 13 GB | 6.5 GB |
| **30B / 32B** | 120 GB | 60 GB | 30 GB | 15 GB |
| **70B** (Llama-3-70B) | 280 GB | 140 GB | 70 GB | 35 GB |

### B. Full fine-tuning with AdamW

Total ≈ (weights + gradients + optimizer states) + activations. Add 10–30% for activations depending on batch size and sequence length.

| Model | Full FP32 (~16 B/p) | Mixed BF16 (~18 B/p) | Realistic GPU setup |
|---|---|---|---|
| **1.5B** | ~24 GB | ~27 GB | 1× RTX 4090 (24 GB) — tight |
| **3B** | ~48 GB | ~54 GB | 1× A100 80 GB |
| **7B** | ~112 GB | ~126 GB | 2× A100 80 GB |
| **8B** | ~128 GB | ~144 GB | 2× A100 80 GB **or** 2× H100 80 GB |
| **13B** | ~208 GB | ~234 GB | 4× A100 80 GB |
| **30B** | ~480 GB | ~540 GB | 8× A100 80 GB |
| **70B** | ~1,120 GB | ~1,260 GB | 8× H100 80 GB (multi-node, ZeRO-3 / FSDP) |

### C. LoRA fine-tuning (16-bit base model + trainable adapters)

The base model is frozen in BF16; only the tiny `A` and `B` matrices receive gradients and Adam states. Memory ≈ base model in BF16 + small adapter overhead + activations.

| Model | LoRA VRAM (approx) | Realistic GPU |
|---|---|---|
| **1.5B** | ~5–6 GB | 1× RTX 3060 (12 GB) |
| **3B** | ~10–12 GB | 1× RTX 3090 / 4090 (24 GB) |
| **7B** | ~20–24 GB | 1× RTX 4090 (24 GB) or A5000 |
| **8B** | ~24–28 GB | 1× RTX 4090 or RTX A6000 (48 GB) |
| **13B** | ~35–40 GB | 1× RTX A6000 (48 GB) or A100 40 GB |
| **30B** | ~70–80 GB | 1× A100 80 GB |
| **70B** | ~160–180 GB | 2× A100 80 GB |

### D. QLoRA fine-tuning (4-bit NF4 base + BF16 adapter)

The base model is quantized to 4-bit (NormalFloat-4), shrinking the largest memory consumer by ~75%. Only the LoRA adapters (a few million params) carry gradients and optimizer state. **This is the practical default for single-GPU fine-tuning in 2026.**

| Model | QLoRA VRAM (approx) | Realistic GPU |
|---|---|---|
| **1.5B** | ~3–4 GB | 1× RTX 3060 (8–12 GB) |
| **3B** | ~5–6 GB | 1× RTX 3060 (12 GB) |
| **7B** | ~6–8 GB | 1× RTX 3060 (12 GB), or RTX 4060 8 GB (with Unsloth) |
| **8B** | ~8–10 GB | 1× RTX 4070 (12 GB) |
| **13B** | ~12–15 GB | 1× RTX 3090 / 4090 (24 GB) |
| **30B** | ~24–28 GB | 1× RTX 4090 / A5000 |
| **70B** | **~46–48 GB** | 1× RTX A6000 (48 GB) or A100 80 GB |

> **The headline result:** QLoRA reduces 70B fine-tuning from a multi-node 8× H100 cluster (~1.2 TB) to a **single 48 GB GPU**. That's a ~25× reduction with only a small quality gap (LoRA typically recovers 90–95% of full fine-tuning quality; QLoRA recovers 80–90%).

---

## The core idea — low-rank decomposition (the math)

Here's the key insight. A weight matrix **W** has shape `d × d` (say 4096 × 4096 = 16M params). When the model adapts to a new task, the **change** in W, called **ΔW**, doesn't need all 16M degrees of freedom. It can be approximated as:

> **ΔW = B × A**

Where:
- **A** has shape `d × r` — initialized with random Gaussian
- **B** has shape `r × d` — initialized with **zeros** (so at the start, ΔW = 0, preserving pre-trained behavior)
- **r** is the *rank* — a tiny number like 4, 8, 16, or 64

The total trainable parameters = `d×r + r×d = 2dr`, which is **orders of magnitude smaller** than `d²`.

The **forward pass** during inference becomes:

> **h = Wx + (B × A)x × (α/r)**

Where `α` is a scaling hyperparameter (controls how much the adapter influences the output).

### Why does this collapse the memory bill?

For an 8B model with LoRA `r=16` applied to Q and V projections across all transformer layers, you're typically training **~10–20 million** parameters (~0.2% of the model). The gradient + AdamW overhead now scales with **20M, not 8B**:

- Trainable params: ~20M
- Gradients (BF16): 20M × 2 = ~40 MB
- AdamW states (FP32): 20M × 8 = ~160 MB
- **Total adapter overhead: ~200 MB** instead of the ~128 GB needed for full fine-tuning.

The frozen 8B base model still sits in VRAM (16 GB in BF16), but it has *no* gradients or optimizer states attached to it. That's the entire trick.

---

<img width="752" height="440" alt="image" src="https://github.com/user-attachments/assets/13993ea8-e726-48ee-b100-fac4c9081bba" />

---

## Where exactly are the LoRA adapters inserted?

LoRA adapters are typically inserted into the **attention layers** of the transformer — specifically into the `Q` (Query) and `V` (Value) projection matrices. Some implementations also target `K`, the output projection, and even the MLP layers. The original W stays completely **frozen**.

**Common target module choices:**
- **Minimal (memory-friendly):** `q_proj`, `v_proj` — original LoRA paper recommendation
- **Standard (2026 default):** `q_proj`, `k_proj`, `v_proj`, `o_proj` — attention block fully
- **Aggressive (best quality):** all linear layers including `gate_proj`, `up_proj`, `down_proj` in the MLP — Unsloth's `target_modules="all-linear"` default

More target modules → more trainable params → slightly more VRAM but typically better task adaptation.

---

<img width="732" height="525" alt="image" src="https://github.com/user-attachments/assets/67f33221-d01f-4813-9fa4-ed92b942d9e9" />

---

## The LoRA training process — step by step

Here's the complete workflow from start to deployment:

---

<img width="787" height="582" alt="image" src="https://github.com/user-attachments/assets/cc2839df-4b5a-497f-a368-addedf7b4723" />

## When do you actually use LoRA?

Here's a decision guide for your sessions:

| Scenario | Use LoRA? | Why |
|---|---|---|
| Adapt GPT/Llama to a specific domain (medical, legal) | ✅ Yes | Classic use case |
| Teach the model a new output format or task style | ✅ Yes | Great fit |
| You have < 100GB VRAM | ✅ Yes | Main motivation |
| Multi-task: serve 10 different clients on one base model | ✅ Yes | Swap adapter per client |
| You need to inject entirely new factual knowledge | ⚠️ Partial | Better with RAG + LoRA |
| Tiny task with < 500 samples | ⚠️ Careful | Risk of overfitting |
| Change model architecture fundamentally | ❌ No | Full pre-training needed |

---

## Recommended hyperparameters (a starting point)

These are the defaults that the 2026 ecosystem (Unsloth, Axolotl, TRL) has converged on:

| Hyperparameter | Default | Notes |
|---|---|---|
| **Rank `r`** | 16 | Higher = more capacity. 8 is fine for style; 32–64 for harder tasks |
| **Alpha `α`** | 16 or 32 | Common convention: α = r (effective scale = 1) or α = 2r |
| **Dropout** | 0.05 | Lower for small datasets, 0.1 for larger ones |
| **Target modules** | `all-linear` | Q/V only saves memory but loses ~3–5% quality |
| **Learning rate** | 2e-4 | 10× higher than full fine-tuning since fewer params |
| **Optimizer** | `paged_adamw_8bit` | 8-bit Adam states cut optimizer memory by ~75% |
| **Batch size** | 1–4 (effective 16–32 via grad accumulation) | Larger batches stabilize training |
| **Epochs** | 1–3 | More overfits adapters fast |

---

## The merge trick — zero inference cost

This is one of LoRA's most elegant properties. After training, you can **merge** the adapters back into the frozen weights:

> **W_new = W + (B × A) × (α/r)**

Now your model is just a normal model again — no adapter overhead, same inference speed, smaller deployment footprint. You can also **not merge** and keep the adapter separate, which lets you hot-swap adapters at runtime for multi-tenant systems (e.g. one base model, different LoRA adapters per enterprise client).

**A practical multi-tenant deployment looks like:**
- 1× base model in BF16 sitting in VRAM (16 GB for 8B)
- N× LoRA adapters (~50–200 MB each) loaded on demand
- Serve hundreds of customer-specific fine-tunes from one GPU

---

## The broader LoRA family — what you'll encounter

Once you understand base LoRA, these variants are easy to grasp:

**QLoRA** — LoRA + 4-bit quantization of the base model. Lets you fine-tune a 70B model on a single 48GB GPU. The quantization is NF4 (Normal Float 4-bit). This is the go-to for resource-constrained settings.

**LoRA+** — Uses different learning rates for A and B matrices. B learns faster than A, which empirically improves convergence.

**DoRA** (Weight-Decomposed LoRA) — Decomposes W into magnitude and direction components, applies LoRA only to the direction. Better stability on some tasks; now enabled by default in Unsloth.

**LoftQ** — Combines quantization and LoRA initialization more carefully for better accuracy when starting from quantized weights.

**rsLoRA** (Rank-Stabilized LoRA) — Uses a scaling factor of `α/√r` instead of `α/r`. This stabilizes training at higher ranks (r ≥ 64), letting you push capacity without divergence.

**VeRA** (Vector-based Random Adaptation) — Shares frozen random `A` and `B` matrices across layers and only trains small scaling vectors. Drastically fewer trainable params at a small quality cost.

**X-LoRA** — Mixture-of-experts-style routing across multiple LoRA adapters, useful when one model needs to serve very different task families.

---
