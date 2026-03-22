# LoRA-Fine-Tuning

---

## Why does LoRA even exist? The problem first.

Imagine you want to fine-tune **Llama 3 (8B parameters)**. Full fine-tuning means updating all 8 billion weights. That needs ~120GB of GPU VRAM just for the gradients and optimizer states — completely out of reach for most teams.

LoRA asks a brilliant question: **do we really need to update all those parameters?**

The answer, backed by research, is **no** — because when models adapt to a new task, the *actual change* in weight matrices tends to live in a **very low-rank space**.---

<img width="736" height="375" alt="image" src="https://github.com/user-attachments/assets/595c4653-4e15-4a56-a398-e2fc0f9fa097" />


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

Where `α` is a scaling hyperparameter (controls how much the adapter influences the output).---

<img width="752" height="440" alt="image" src="https://github.com/user-attachments/assets/13993ea8-e726-48ee-b100-fac4c9081bba" />


## Where exactly are the LoRA adapters inserted?

LoRA adapters are typically inserted into the **attention layers** of the transformer — specifically into the `Q` (Query) and `V` (Value) projection matrices. Some implementations also target `K`, the output projection, and even the MLP layers. The original W stays completely **frozen**.---

<img width="732" height="525" alt="image" src="https://github.com/user-attachments/assets/67f33221-d01f-4813-9fa4-ed92b942d9e9" />


## The LoRA training process — step by step

Here's the complete workflow from start to deployment:---

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

## The merge trick — zero inference cost

This is one of LoRA's most elegant properties. After training, you can **merge** the adapters back into the frozen weights:

> **W_new = W + (B × A) × (α/r)**

Now your model is just a normal model again — no adapter overhead, same inference speed, smaller deployment footprint. You can also **not merge** and keep the adapter separate, which lets you hot-swap adapters at runtime for multi-tenant systems (e.g. one base model, different LoRA adapters per enterprise client).

---

## The broader LoRA family — what you'll encounter

Once you understand base LoRA, these variants are easy to grasp:

**QLoRA** — LoRA + 4-bit quantization of the base model. Lets you fine-tune a 70B model on a single 48GB GPU. The quantization is NF4 (Normal Float 4-bit). This is the go-to for resource-constrained settings.

**LoRA+** — Uses different learning rates for A and B matrices. B learns faster than A, which empirically improves convergence.

**DoRA** — Decomposes W into magnitude and direction components, applies LoRA only to the direction. Better stability on some tasks.

**LoftQ** — Combines quantization and LoRA initialization more carefully for better accuracy when starting from quantized weights.

---

