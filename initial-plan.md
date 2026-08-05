# LLM Engineering: A Self-Paced Phase Plan

Built from the course syllabus, restructured so progress is gated by **what you can do**, not by what week it is.

**Hardware this plan assumes:** MacBook Pro M4, 16GB unified memory, plus Kaggle's free GPU tier. No local NVIDIA GPU. See [Your Hardware Setup](#your-hardware-setup) — it's load-bearing, read it before Phase 0.

---

## How this works

Seven phases, plus two tracks that run continuously underneath them.

Each phase has an **exit gate**: a concrete thing you must be able to build or explain before moving on. The gate is the whole point. A week can pass whether or not you learned anything; a gate can't.

**Rules for yourself:**
- You may not start a phase until the previous gate is cleared. No exceptions for phases 1–4.
- Effort ranges below assume ~10 hrs/week, but ignore them — they're for budgeting, not pacing. Some people clear Phase 2 in ten days, some in six weeks. Both are fine.
- If you stall for more than two sessions on one concept, ship a broken version, write down exactly what confuses you, and move on. Come back after the next phase. Understanding often arrives sideways.

**The four-step loop, every single unit:**
1. **Theory** — one lecture or chapter, once, at 1.5x, no notes.
2. **Reimplement** — write it from scratch, badly, without looking. Then look, then fix.
3. **Measure** — every claim in this field is a performance claim. Tokens/sec, VRAM, perplexity, accuracy. Numbers or it didn't happen.
4. **Write** — a short README explaining it to someone one phase behind you.

**The one sentence tying the whole syllabus together:**
> A transformer is a stack of matrix multiplies. Training is bandwidth-bound at scale, inference is memory-bound at small batch. Almost everything downstream is a trick to move fewer bytes.

---

## Your Hardware Setup

There is **no CUDA on Apple Silicon**. Not "difficult" — no path at all. No NVIDIA driver on modern macOS, no CUDA for ARM Macs, no Metal backend for Triton. Don't spend a weekend on it.

This matters less than it sounds. The work splits cleanly, and the M4 is genuinely good at one half.

| Do locally on the M4 | Needs an NVIDIA GPU (Kaggle) |
|---|---|
| Roofline reasoning, arithmetic intensity, bandwidth math | CUDA and Triton kernel writing |
| Reading — PMPP, CS336, GPU MODE, all papers | Multi-GPU: DDP, FSDP, NCCL, tensor parallel |
| Quantization experiments (llama.cpp is native here) | Any training run of consequence |
| LoRA fine-tuning of small models via MLX | vLLM / TensorRT-LLM serving |
| Edge deployment — all of Phase 6 Branch B | Anything needing >12GB of accelerator memory |
| Debugging every training script before it costs quota | |

**Your M4's numbers, which you will use constantly:** 120 GB/s memory bandwidth, 16GB unified. macOS reserves memory conservatively — you get ~10.6GB for GPU by default. Raise it with:

```bash
sudo sysctl iogpu.wired_limit_mb=12288
```

Realistic local model sizes: 8B at 4-bit is comfortable, 14B at 4-bit works with other apps closed, 32B is out.

### Kaggle free tier — your GPU

- **~30 GPU-hours per week**, resets weekly. This is a lot; the entire plan's GPU-dependent work is maybe 60–80 hours.
- **1× P100 (16GB, ~732 GB/s, no tensor cores)** or **2× T4 (16GB each, ~320 GB/s, FP16 tensor cores)**. Pick deliberately — don't accept the default.
- Sessions run up to 12 hours. 20GB persistent storage.
- **Phone-verify your account first** — GPU access is gated behind it.
- Put a GitHub personal access token in **Kaggle Secrets** so notebooks can push results without re-pasting credentials.

**Writing CUDA in a Kaggle notebook** — `nvcc` is in the image:

```python
!pip install nvcc4jupyter
%load_ext nvcc4jupyter
# then use %%cuda at the top of a cell
```

Or manually: `%%writefile kernel.cu`, then `!nvcc -arch=sm_75 -o kernel kernel.cu && ./kernel`. Use `sm_75` for T4, `sm_60` for P100. Triton works on T4 as well.

### What Kaggle cannot do

Know these before you lose a day to them:

- **FlashAttention 2 will not run.** Requires Ampere or newer (sm_80+); T4 is sm_75. Read the paper, study the tiling, but you can't run the official kernels. → rent once.
- **No bf16 on Turing or Pascal.** Use fp16 and expect occasional loss instability. Many modern recipes assume bf16 and will need edits. Annoying, but you'll learn exactly why bf16 exists.
- **Nsight Compute profiling is blocked** — `ncu` needs elevated GPU counter permissions hosted notebooks don't grant. Use `torch.profiler`, CUDA events, and `torch.cuda.max_memory_allocated()` instead. For real kernel profiling you need root → rent.
- **No FP8, no Hopper features.** Skip; read about them.
- **vLLM/TensorRT-LLM benchmarks run but aren't representative** of modern serving hardware. Fine for mechanics; don't publish the numbers as meaningful.

### Renting, for the few gaps

Total rented time across the entire plan: roughly **two 6-hour sessions**, ≈₹800.

- **Vast.ai / RunPod** spot — a 4090 or L4 at ₹30–60/hr, with root, so Nsight works. Use for the FlashAttention 2 session and the Nsight session.
- **Modal** — best fit for a Mac user: write Python locally, decorate a function, it executes on a cloud GPU. No SSH, no environment drift. Has monthly free credits.
- **Colab** free tier as overflow when Kaggle quota runs out.

**Do not buy hardware for this.** Rent by the hour, and only for what's listed above.

### Quota discipline

The default failure mode is burning 30 hours on nothing.

- **Debug on the Mac first.** Run every training loop on CPU with a 2-layer model and batch size 2 until one step completes cleanly. *Then* go to Kaggle. This alone saves half your quota.
- **Use "Save Version → Run All"** for long jobs — it runs in the background and stops cleanly instead of you paying to watch a notebook.
- **Kill idle sessions manually.** Idle burns quota.
- **Checkpoint to `/kaggle/working` and push to GitHub or a Kaggle Dataset every epoch.** Session state does not survive.

---

## The two continuous tracks

These don't have phases. They start early and never stop.

### Track A — Evaluation
The original syllabus puts evaluation near the end. **Don't wait.** Start building eval habits from Phase 3 onward. Every deliverable from Phase 3 on must answer "did it get better, and how do I know?"

Formal study (harnesses, LLM-as-judge, observability) lands in Phase 5, but the habit starts the moment you have two versions of anything to compare.

- **AI Engineering** (Chip Huyen) Ch. 3–4 — the best treatment anywhere. Read it early even if you don't follow all of it yet.
- EleutherAI [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness/), [Hamel Husain's writing on evals.](https://hamel.dev/blog/posts/evals-faq/)
- Build your own tiny harness in Phase 3 and reuse it in every later phase.

Evals are the actual moat. Anyone can call an API. The person who knows whether the output got better is the engineer.

### Track B — Papers & Writing
~2 papers per week regardless of phase. Three-pass method: skim structure → read figures and results → read methods only if you'll implement it. Papers are Mac work; no GPU needed.

One public write-up per phase. Blog, dev.to, or long-form README.

**The canon (~30 papers, the whole plan):**

| Cluster | Papers |
|---|---|
| Foundations | Attention Is All You Need · GPT-2 · GPT-3 · Chinchilla · LLaMA 3 · Switch Transformer · Mixtral · DeepSeek-V3 |
| Systems | FlashAttention 2 · PagedAttention (vLLM) · GQA · Megatron-LM · ZeRO · Speculative Decoding |
| Compression | LLM.int8() · GPTQ · AWQ · QLoRA · SmoothQuant |
| Training & alignment | LoRA · InstructGPT · DPO · DeepSeek-R1 (GRPO) · LIMA |
| Applications | RAG · ReAct · Chain-of-Thought · Self-Consistency · Toolformer · CLIP · LLaVA |
| Frontier | Mamba-2 · Jamba |

---

## Core library

Five books. You don't need twenty.

| Book | Author | Phase |
|---|---|---|
| **Build a Large Language Model (From Scratch)** | Sebastian Raschka | 1, 4 |
| **Hands-On Large Language Models** | Alammar & Grootendorst | 1, 5 |
| **AI Engineering** | Chip Huyen | 4, 5 — and Track A |
| **Programming Massively Parallel Processors, 4th ed.** (Ch. 1–6 only) | Hwu, Kirk, El Hajj | 2 |
| **LLM Engineer's Handbook** | Iusztin & Labonne | 4, 5 |

**Free, treat as required:**
- **Speech and Language Processing, 3rd ed.** — Jurafsky & Martin (free draft)
- **The Ultra-Scale Playbook** — Hugging Face. Best free text on multi-GPU/multi-node parallelism.
- **How to Scale Your Model** — Google DeepMind. Roofline thinking, arithmetic intensity.
- **RLHF Book** — Nathan Lambert (free online)

**Anchor courses:**
- **Stanford CS336 — Language Modeling from Scratch** (Spring 2026 lectures free on YouTube, assignments public). Covers Phases 1–3 almost exactly.
- **Karpathy — Neural Networks: Zero to Hero** + nanoGPT + nanochat. Non-negotiable for Phase 1.
- **MIT 6.5940 — TinyML and Efficient Deep Learning Computing** (Song Han). Owns Phase 3 and the edge work in Phase 6.
- **GPU MODE** lecture series + Discord. Phase 2 — and their notebooks are built for hosted environments like Kaggle/Colab, which suits you exactly.
- **Umar Jamil (YouTube)** — from-scratch LLaMA, LoRA, DPO, Flash Attention, quantization.

Blogs: Sebastian Raschka (*Ahead of AI*), Lilian Weng, Jay Alammar, HF blog, vLLM blog, Simon Willison.

---

# PHASE 0 — Ground Floor
*Effort: 1–2 sessions. Do it today.*

Get the scaffolding up before you learn anything, so there's never friction between understanding something and recording it.

- Create the repo, write the root README with an empty gate table, commit it.
- Phone-verify Kaggle. Add a GitHub PAT to Kaggle Secrets.
- Local: install `uv` or conda, PyTorch with MPS, MLX (`pip install mlx mlx-lm`), and llama.cpp. Confirm `mlx_lm.generate` runs an 8B model at 4-bit.
- Raise the Metal wired-memory limit (see Hardware Setup).
- Set up `notes/` and `benchmarks/` folders.

### 🚪 Exit gate
**Two things, both from muscle memory in under five minutes each:**
1. Open a Kaggle notebook, attach a GPU, run `nvidia-smi`, and `git push` a result back to your repo.
2. Run a quantized 8B model locally on the Mac and print tokens/sec.

---

# PHASE 1 — Foundations
### *You understand what the model actually is*
*Effort: 3–6 weeks · Syllabus themes: LLM Foundations I & II*
*Compute: one T4 is fully sufficient. Much of this is Mac work.*

The goal is not "I've seen a transformer diagram." It's that the architecture holds no mystery for you.

### Unit 1.1 — Transformer architecture
*Attention Is All You Need* · GPT-1 & GPT-2 papers · Alammar's "Illustrated Transformer" · Karpathy "Let's build GPT" · Raschka Ch. 1–4 · read nanoGPT's `model.py` line by line.

### Unit 1.2 — Tokenization & pretraining objectives
Karpathy "Let's build the GPT Tokenizer" · CS336 Lecture 1 · HF Tokenizers docs · BPE, SentencePiece, tiktoken.

> Don't skip tokenization because it looks boring. Half of all weird LLM behaviour — arithmetic failures, non-English degradation, injection edge cases — traces back to the tokenizer. Especially relevant for Indic scripts, where token efficiency is brutal.

### Unit 1.3 — Mixture of Experts
Switch Transformer · Mixtral · DeepSeekMoE / DeepSeek-V3 · HF "Mixture of Experts Explained".

### Unit 1.4 — Modern architectures & scaling laws
Raschka's "Big LLM Architecture Comparison" posts · LLaMA 1/2/3 · Qwen technical reports · Gemma · OLMo (read the *data* section — nobody else publishes it) · Kaplan 2020 · Chinchilla · *Are Emergent Abilities a Mirage?* (read as a corrective).

### Deliverables
- `01-transformer-from-scratch/` — your own GPT-2 (~124M) in one file, plus your own BPE tokenizer. Train on TinyStories. Report loss curve and samples. **Write and debug it on the Mac with MPS at tiny scale, then run the real training on Kaggle.** That workflow is the one you'll use for the rest of the plan; establish it now.
- `02-architecture-atlas/` — a table you maintain: for 8–10 open models, log norm placement, positional encoding (RoPE variants), attention type (MHA/GQA/MQA/MLA), FFN activation, dense vs MoE, vocab size, context length. Pure Mac work.
- One scaling-law experiment: train 4 tiny models at increasing size on fixed data, plot loss vs params log-log. Watching the straight line appear in *your* numbers changes how you read the field. Budget ~4 GPU-hours.

### 🚪 Exit gate
**Write a transformer forward pass from memory on a blank page** — shapes included, no reference. Then explain out loud, to nobody, why GQA exists and what it trades away. If you hesitate on either, you're not done.

---

# PHASE 2 — The Hardware Layer
### *You understand what the model runs on*
*Effort: 3–5 weeks · Syllabus theme: GPU Basics*
*Compute: ~25–40 Kaggle GPU-hours, plus one rented session for Nsight.*

**This is the wall.** It's the hardest, least glamorous, most valuable part of the syllabus, and it's where most self-learners quietly quit and stay at the "I call APIs" level permanently. Everything in Phase 3 is incomprehensible without it and trivial with it.

Do not skip this phase because you lack a local GPU. Roughly half of it is arithmetic you do on paper, and your M4 is a *better* teaching instrument for that half than a rented A100, because its constraints are tight enough to be visible.

### Unit 2.1 — GPU architecture *(mostly Mac)*
PMPP Ch. 1–6 · CS336 "GPUs, TPUs" lecture · Horace He's "Making Deep Learning Go Brrrr From First Principles" · CUDA C++ Programming Guide (skim only).

Optional but valuable bridge: write a few **Metal** kernels locally. The mental model maps almost one-to-one — threadgroups ≈ blocks, SIMD groups ≈ warps, threadgroup memory ≈ shared memory, identical coalescing and occupancy reasoning. MLX exposes custom Metal kernels from Python (`mlx.core.fast.metal_kernel`), so you can do naive matmul → tiled → shared-memory-optimized without leaving a notebook. When you write the CUDA version later you'll be translating syntax, not learning concepts.

### Unit 2.2 — Parallelism *(Kaggle, 2× T4)*
**Ultra-Scale Playbook** (primary text) · Megatron-LM · ZeRO/DeepSpeed · PyTorch FSDP docs · NCCL collectives. Understand data vs tensor vs pipeline vs expert parallelism well enough to say which you'd reach for and why.

The **T4 × 2** option gives you real DDP, FSDP sharding, NCCL collectives, and toy tensor parallelism, free. Bonus lesson baked in: those two T4s talk over PCIe with **no NVLink**, so communication overhead is visible. Measure scaling efficiency at 1 vs 2 GPUs and you'll understand why interconnect topology gets its own section in every cluster design doc.

### Unit 2.3 — The physical stack *(Mac)*
H100/H200/B200 specs — specifically memory bandwidth vs FLOPs, and note how that ratio keeps getting worse · NVLink vs PCIe vs InfiniBand · Semianalysis posts on cluster design.

### Deliverables

**A. `03-gpu-fundamentals/roofline-across-three-chips/` — the headline artifact.**

You have free access to three accelerators with sharply different profiles:

| | Bandwidth | Tensor cores |
|---|---|---|
| M4 (local) | 120 GB/s | No |
| Kaggle T4 | ~320 GB/s | Yes (FP16) |
| Kaggle P100 | ~732 GB/s | No |

The P100 has more than twice the T4's bandwidth but no tensor cores. **Predict first, then measure:** decode throughput should track bandwidth (P100 wins); prefill and matmul-heavy training steps should favour the T4's tensor cores.

Same model, same code, three chips. This demonstrates memory-bound vs compute-bound more convincingly than any lecture, clears the exit gate outright, and costs nothing. It's also an unusual portfolio piece — nobody benchmarks a P100 against an M4 on purpose.

**B. The local bandwidth demo.** Load an 8B model at Q4_K_M (~4.7GB) in llama.cpp on the Mac. Predict decode speed before running: 120 ÷ 4.7 ≈ **25 tok/s ceiling**. Measure — you'll see ~18–22. Then go to Q8 (~8.5GB), predict ~14 tok/s, confirm. The model got no smarter and got slower in exact proportion to its size in bytes. Ten minutes, no GPU, whole lesson delivered.

**C. A roofline calculator.** Given a model config, batch size, and a chip's specs, predict arithmetic intensity and whether the workload is memory- or compute-bound. Validate against A and B.

**D. Three Triton kernels** on Kaggle: vector add, fused softmax, naive matmul. Benchmark against PyTorch and explain the gap.

### 🚪 Exit gate
**Given an arbitrary model config, batch size, and chip, predict on paper whether the workload is memory-bound or compute-bound — then run it and be right.** Twice, on different chips. Your three-chip setup makes this easy to test honestly.

---

# PHASE 3 — Inference & Efficiency
### *You can make it fast and small*
*Effort: 4–6 weeks · Syllabus themes: Inference, Efficient Inference & Quantization*
*Compute: heavily Mac-based. One rented session for FlashAttention 2 and a representative serving benchmark.*

This is the highest-value phase for employability — serving skills are scarce. It's also, fortunately, the phase your Mac is best suited to: llama.cpp and GGUF were essentially born on Apple Silicon.

### Unit 3.1 — Sampling & generation
Temperature, top-k, top-p, min-p, beam search · HF `generate` source · *The Curious Case of Neural Text Degeneration*.

### Unit 3.2 — Inference math
**Kipply's "Transformer Inference Arithmetic"** (essential — read it three times) · *How to Scale Your Model* · prefill vs decode · TTFT vs ITL vs throughput · continuous batching.

> The key mental model: during decode you re-read the *entire* model's weights from HBM to produce **one** token. That's why batching gives near-free throughput and why quantization is such a large win. You already proved this on your own hardware in Phase 2.

### Unit 3.3 — Efficient attention & KV caching
FlashAttention 1/2/3 · PagedAttention (vLLM paper) · GQA & MQA · Multi-head Latent Attention (DeepSeek-V2) · sliding window (Mistral).

**Note:** FlashAttention 2 needs sm_80+, so neither your Mac nor Kaggle can run it. Study the tiling and IO-awareness argument from the paper, implement a simple tiled attention yourself, then rent one session to run and profile the real thing.

### Unit 3.4 — Quantization *(Mac-native)*
MIT 6.5940 quantization lectures (best resource available) · LLM.int8() · GPTQ · AWQ · SmoothQuant · QLoRA's NF4 · FP8/FP4 · llama.cpp GGUF quant types.

### Unit 3.5 — Serving engines
vLLM docs + paper · SGLang (RadixAttention) · TensorRT-LLM · llama.cpp · Ollama · speculative decoding (Leviathan et al.) · Medusa · tensor parallelism for serving.

vLLM on macOS is CPU-backend only and not representative. Run the mechanics on Kaggle to learn the API and continuous batching behaviour; run the *publishable* benchmark on a rented L4 or 4090.

### Deliverables
- `04-inference-internals/` — implement KV caching yourself on your Phase 1 GPT-2. Measure tokens/sec with and without. Then write a naive continuous batcher. The gap between yours and vLLM's is the lesson. Runs fine on the Mac.
- `05-quantization-lab/` — **the single most portfolio-valuable artifact in the plan, and it's fully local.** Take one 8B model. Produce FP16, INT8, AWQ-4bit, and GGUF Q4_K_M. For each: memory, tokens/sec, and score on a small eval set. One chart, four points, real conclusions. Extend it by re-running the same four variants on T4 and P100 — now it's a quantization × hardware matrix, which almost nobody publishes.
- Serving comparison: llama.cpp on Mac vs vLLM on Kaggle vs SGLang on a rented GPU. Report p50/p99 under load, not averages, and label the hardware on every number.
- **Track A starts here:** build the reusable eval harness you'll plug into every later phase.

### 🚪 Exit gate
**Take a model that "doesn't fit" on your 16GB Mac, get it serving locally, and produce a defensible chart of the quality-vs-speed-vs-memory tradeoff you made.** If someone asks "why AWQ over GPTQ here?" you have an answer grounded in your own numbers.

---

# PHASE 4 — Training & Adaptation
### *You can change what the model does*
*Effort: 4–7 weeks · Syllabus theme: Fine-Tuning Fundamentals*
*Compute: QLoRA on an 8B model fits one T4 with Unsloth. Kaggle is sufficient throughout.*

**Workflow for this phase:** build and debug the whole pipeline locally with MLX-LM, which supports LoRA fine-tuning of 7–8B models on your M4 — slow but real. Once one step runs clean, do the actual training run on Kaggle with Unsloth. Never debug on rented or quota'd compute.

Watch for the bf16 gap: T4 and P100 are fp16-only, so recipes assuming bf16 need edits and may show loss instability. Note it in your write-up; it's a real production consideration, not just an inconvenience.

### Unit 4.1 — Full FT vs PEFT
LoRA · QLoRA · DoRA · PEFT survey · Raschka's LoRA-from-scratch posts · **AI Engineering Ch. 7** for the prior question: *should you fine-tune at all?* (Usually: no — try prompting and RAG first.)

### Unit 4.2 — Instruction tuning
InstructGPT · FLAN · Self-Instruct · LIMA (*less is more* — quality beats quantity) · Alpaca/Dolly formats · **chat templates**, which is where most of your bugs will actually live.

### Unit 4.3 — Preference alignment
InstructGPT (RLHF) · DPO · KTO · ORPO · SimPO · HF TRL docs · Nathan Lambert's RLHF Book.

### Unit 4.4 — RL for LLMs
PPO basics · GRPO (DeepSeek-R1 / DeepSeekMath) · RLVR (verifiable rewards) · reward models and LLM-as-judge rewards · reward hacking.

**Tooling:** Unsloth (fastest single-GPU path, runs well on free Kaggle T4s), TRL, PEFT, Axolotl or LLaMA-Factory for config-driven runs, MLX-LM for local iteration.

### Deliverable
`06-finetuning/` — a real fine-tune on a domain you care about. Full loop: dataset construction → LoRA SFT → DPO or GRPO pass → eval against base → merge → GGUF export → **run it locally on the Mac.** That last step closes the loop nicely: you trained it on a T4 and you're serving it on your laptop.

**Document the failures.** The loss spike, the wrong chat template, the fp16 overflow, the model that learned verbosity instead of correctness. That write-up is worth more than the model.

> Hard truth to write on a sticky note now: most fine-tuning projects fail on **data**, not method. Budget 60% of this phase on the dataset — and that part costs zero GPU-hours.

### 🚪 Exit gate
**Ship a fine-tuned model that measurably beats its base model on an eval set you built yourself — and be able to explain why the improvement is real and not eval contamination or a judge artifact.**

---

# PHASE 5 — Systems on Top
### *You can build things people use*
*Effort: 5–8 weeks · Syllabus themes: Reasoning, RAG, Agents, Tool Use, Agent Fine-Tuning, Evaluation*
*Compute: almost entirely Mac and API work. Barely touches your quota.*

The largest phase and the one most tutorials start with. You're arriving with foundations, which means you'll build these an order of magnitude better than the average practitioner.

### Unit 5.1 — Reasoning
Chain-of-Thought (Wei et al.) · Zero-shot CoT · Self-Consistency · Tree of Thoughts · Least-to-Most · DeepSeek-R1 (the RL-for-reasoning turning point) · o-series system cards · test-time compute scaling · **DSPy** (the real answer to "prompting as code").

*Deliverable:* `07-reasoning/` — on a reasoning benchmark subset, compare direct → CoT → self-consistency (n=8) → DSPy-optimized. **Plot accuracy against token cost, always.** That framing is what makes it engineering rather than prompt collecting.

### Unit 5.2 — RAG
Original RAG paper · **AI Engineering Ch. 6** · *Hands-On LLMs* Ch. 8 · chunking strategies · Anthropic's Contextual Retrieval post · hybrid search (BM25 + dense) · ColBERT/ColPali late interaction · cross-encoder rerankers · RAGAS · retrieval metrics (recall@k, MRR, nDCG) vs generation metrics.

Build bare-metal first — embedding model + FAISS/Qdrant, all of which run happily on the M4 — *then* look at LlamaIndex or Haystack, so you know what the framework hides.

*Deliverable:* `08-rag/` — RAG over a corpus you know well enough to catch it lying. Write 50 hand-made Q&A pairs **before** building the pipeline. Then ablate: naive vs semantic chunking, dense-only vs hybrid, ± reranker, ± contextual retrieval.

> The thing most people miss: RAG failures are usually *retrieval* failures, not generation failures. Instrument retrieval separately or you'll spend weeks tuning prompts to fix a chunking bug.

### Unit 5.3 — Agents & tool use
ReAct paper · **Anthropic's "Building Effective Agents"** (read it twice) · Toolformer · MCP spec at modelcontextprotocol.io · Self-RAG · CRAG · LangGraph · the honest literature on when multi-agent *hurts*.

*Deliverable:* `09-agents/` — write the agent loop from scratch first (~150 lines: LLM call → parse tool call → execute → append observation → repeat). Then rebuild with LangGraph and write up what the framework bought and what it cost. Then ship one MCP server that does something genuinely useful to you.

### Unit 5.4 — Fine-tuning for tool use
ToolLLM/ToolBench · xLAM · Gorilla · function-calling dataset formats · training on trajectories. Optional if Phase 4 exhausted you — come back to it.

### Unit 5.5 — Evaluation & observability (Track A goes formal)
**AI Engineering Ch. 3–4** · lm-evaluation-harness · HELM · benchmark contamination · LLM-as-judge failure modes (position bias, verbosity bias, self-preference) · agent benchmarks (τ-bench, AgentBench, SWE-bench, WebArena) · Langfuse (open source, self-hostable — runs locally in Docker on the M4), Arize Phoenix, LangSmith · OpenTelemetry GenAI semantic conventions.

*Deliverable:* `10-eval-harness/` — mature your Phase 3 harness into something reusable. Self-host Langfuse and trace your Unit 5.2 RAG app end to end.

### 🚪 Exit gate
**A deployed system someone other than you has used, with a tracing dashboard and an eval suite that catches regressions before users do.**

---

# PHASE 6 — Specialization
### *Pick two. Skip the rest without guilt.*
*Effort: 3–5 weeks per branch · Syllabus themes: Multimodal, Edge, Security, Frontiers*

This is the branch point. Four directions, all legitimate. Depth in two beats a tour of four.

**Given your hardware, Branch B is the natural pick** — a 16GB Apple Silicon laptop is the ideal development machine for on-device inference, and you'll have spent Phases 2–4 building exactly the intuitions it needs. Branch C is the cheapest second pick (almost no GPU required).

### Branch A — Multimodal *(needs the most GPU of the four)*
CLIP · **LLaVA** (start here, simplest architecture to grasp) · Flamingo (cross-attention approach) · Qwen-VL / Qwen-Omni reports · Whisper (runs great locally via whisper.cpp) · vision encoders and projector layers · Unsloth VLM support.

*Deliverable:* `11-multimodal/` — fine-tune a small VLM on narrow document understanding: invoices, forms, handwritten notes, regional-language signage. **Multimodal + Indic-script documents is genuinely underserved** and makes a distinctive portfolio piece. Budget most of a week's quota.

### Branch B — Edge *(your strongest branch — nearly all local)*
**MIT 6.5940 is the home turf here** · Phi series · Gemma small variants · Qwen 0.5B/1.5B · MobileLLM · SmolLM · distillation · structured pruning · llama.cpp internals · GGUF · ExecuTorch · MLC-LLM · ONNX Runtime · Core ML and the Apple Neural Engine · NPU/mobile inference.

*Deliverable:* `12-edge/` — run a model on the most constrained device you can reach: Android via MLC-LLM or Termux, or a Raspberry Pi. Report tokens/sec, memory, thermal throttling, battery drain. Then add your Mac as a data point — the M4's 120 GB/s vs a phone's ~50 GB/s is the same roofline story one tier down. Very few people actually do this and it shows immediately.

### Branch C — Security & Privacy *(near-zero GPU)*
**OWASP Top 10 for LLM Applications** (the canonical checklist) · Simon Willison's prompt-injection archive (the definitive ongoing record) · the *lethal trifecta*: private data + untrusted content + external communication · garak and PyRIT for red-teaming · membership inference · training-data extraction · Anthropic and OpenAI red-team publications.

*Deliverable:* `13-security/` — red-team your own Phase 5 RAG app. Attempt indirect prompt injection via a poisoned document. Document what worked, implement mitigations, re-test. Honest security write-ups are rare and hiring managers notice them.

### Branch D — Frontier architectures
Mamba & Mamba-2 (Gu & Dao) · *The Annotated S4* · RWKV · Jamba (hybrid Transformer-Mamba) · linear and hybrid attention · Qwen-Next and recent hybrid releases · diffusion language models.

*Deliverable:* pick **one** architecture, implement a minimal version, write the explainer you wish had existed when you started. Note: several Mamba reference implementations depend on CUDA-only selective-scan kernels, so plan to run on Kaggle; a slow pure-PyTorch version on the Mac is fine for understanding.

> This branch will age fastest. That's fine — the skill you're actually building is *reading a new technical report and correctly placing it*, not memorizing current SOTA.

### 🚪 Exit gate
**Two branches with real, benchmarked, documented artifacts.** Not four half-finished ones.

---

# PHASE 7 — Capstone & Signal
### *You can show it*
*Effort: 4–6 weeks*

Ship one system touching most of the stack:

> A domain-specific assistant — fine-tuned small model, quantized, served on vLLM, hybrid RAG, agentic tool use over MCP, a real eval suite, Langfuse tracing, and a documented cost/latency profile.

**A variant that suits your hardware better and is arguably more interesting:** make it *locally deployable*. Same system, but the whole thing runs on a 16GB laptop — 4-bit model via llama.cpp, local embeddings, local Langfuse. Then publish the cost/latency comparison against the cloud-served version. "I built this to run entirely on-device and here's what it cost me in quality" is a sharper story than another cloud RAG demo, and it's the natural culmination of the constraint you've been working under all along.

Then **present it**. Record a 20-minute walkthrough for YouTube, or give the talk at a meetup — Chennai has active Python and ML groups, and PyCon India takes first-time speakers. Explaining it live exposes every remaining gap, which is exactly the point.

### 🚪 Exit gate
Someone in the field watches your walkthrough and asks you a hard follow-up question — and you enjoy answering it.

---

## Compute at a glance

| Phase | Kaggle free tier | Rent |
|---|---|---|
| 0 Ground floor | Setup only | — |
| 1 Foundations | Fully sufficient — GPT-2 124M trains on one T4 | — |
| 2 Hardware | ~90%: CUDA, Triton, 2× T4 multi-GPU, roofline | One session for Nsight profiling |
| 3 Inference | Mostly Mac; T4/P100 for the hardware matrix | One session for FlashAttention 2 + a representative serving benchmark |
| 4 Adaptation | Fully sufficient — QLoRA on 8B fits one T4 | Optional, for speed |
| 5 Systems | Barely needed; Mac and APIs | — |
| 6 Branch A (multimodal) | Tight but workable | Likely one session |
| 6 Branch B (edge) | Mac is the ideal dev box | — |
| 6 Branch C (security) | Near-zero GPU | — |
| 6 Branch D (frontier) | Needed for CUDA-only kernels | — |
| 7 Capstone | Depends on design | — |

**Total rented: roughly two 6-hour sessions, ≈₹800 across the whole plan.** The hardware was never the constraint.

---

## The minimum viable path

If life compresses and you have to cut: **Phases 1 → 2 → 3 → 4**, plus one branch from Phase 6, is a complete and defensible skill set. Phase 5 content is the most widely taught and the easiest to pick up on the job. Phases 2 and 3 are the opposite — nobody teaches them and almost nobody has them.

If you only ever finish Phase 3, you are already more useful than most people with "LLM" on their résumé.

---

## GitHub structure

One monorepo, plus separate repos for the two or three standouts.

```
llm-engineering-journey/
├── README.md              ← front door: gate table, links, hardware note
├── LEARNINGS.md           ← running list of one-line insights
├── HARDWARE.md            ← your setup, wired-memory tweak, Kaggle workflow, quota log
├── phase-0-setup/
├── phase-1-foundations/
│   ├── 01-transformer-from-scratch/
│   └── 02-architecture-atlas/
├── phase-2-hardware/
│   ├── 03-gpu-fundamentals/
│   └── roofline-across-three-chips/
├── phase-3-inference/
├── phase-4-adaptation/
├── phase-5-systems/
├── phase-6-<your-branches>/
├── notes/                 ← one .md per paper
├── benchmarks/            ← every measurement, reused across phases
└── capstone/
```

**Root README should show a gate table, not a calendar:**

| Phase | Gate | Status | Cleared |
|---|---|---|---|
| 0 | Kaggle GPU + local 8B in 5 min | ✅ | 2026-08-02 |
| 1 | Transformer from memory | 🔄 In progress | — |
| 2 | Predict memory- vs compute-bound | ⬜ | — |
| 3 | Serve a model that "doesn't fit" | ⬜ | — |

That table is more compelling than a commit graph, because it advertises capabilities rather than activity.

**Per-project README skeleton:**

```markdown
# <Title>
**Phase:** 3 · **Status:** Done · **Time:** ~9 hrs
**Hardware:** M4 16GB / Kaggle T4 / rented L4

## What I built
## Why it matters (the engineering problem)
## Results
<table or chart — numbers, hardware, cost, always>
## What surprised me
## What I got wrong first
## References
```

Two sections carry disproportionate weight:

**"What I got wrong first"** — everyone's repo shows the working version. Almost nobody shows the debugging, and that's the part that reads as real experience.

**Hardware on every benchmark** — since you're deliberately working across an M4, a T4, a P100, and occasionally a rented GPU, labelling the chip on every number isn't bookkeeping, it's the substance. *"18 tok/s (M4, Q4_K_M, 120 GB/s) vs 41 tok/s (P100, same quant)"* is a result. *"18 tok/s"* alone is noise.

**Habits:**
- Commit small and often, not one big dump per phase.
- Push negative results. "LoRA rank 64 gave no gain over rank 16 here" is a finding.
- Charts over paragraphs for results. Matplotlib is fine.
- Pin dependencies, and note MPS vs CUDA divergences where they bit you — that's useful to others and rarely documented.
- Log your weekly Kaggle quota spend in `HARDWARE.md`. It'll teach you where your time actually goes.

**High leverage, optional:** publish one fine-tuned model + dataset to HF Hub with a real model card · turn `notes/` into an MkDocs Material site (one afternoon) · land one small PR on vLLM, TRL, Unsloth, MLX, or llama.cpp during Phase 5+, even docs — MLX and llama.cpp are especially approachable from a Mac, and it changes how you read codebases.

---

## Two last warnings

**Don't chase releases.** New models drop weekly and it will feel like falling behind. It isn't. The fundamentals in Phases 1–4 have been stable for three years; model names are noise layered on top.

**Depth beats coverage — and phases make this easy to get wrong in the other direction.** Without week boundaries, the failure mode flips from rushing to drifting. Set yourself a soft ceiling: if a phase passes eight weeks, clear the gate with whatever you have and move. Momentum is a real resource.

Start Phase 0 today, not Monday.
