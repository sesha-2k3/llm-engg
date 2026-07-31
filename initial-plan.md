# LLM Engineering: A Self-Paced Phase Plan

Built from the course syllabus, restructured so progress is gated by **what you can do**, not by what week it is.

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

## The two continuous tracks

These don't have phases. They start early and never stop.

### Track A — Evaluation
The original syllabus puts evaluation near the end. **Don't wait.** Start building eval habits from Phase 3 onward. Every deliverable from Phase 3 on must answer "did it get better, and how do I know?"

Formal study of it (harnesses, LLM-as-judge, observability) lands in Phase 5, but the habit starts the moment you have two versions of anything to compare.

- **AI Engineering** (Chip Huyen) Ch. 3–4 — the best treatment anywhere. Read it early even if you don't understand all of it yet.
- EleutherAI `lm-evaluation-harness`, Hamel Husain's writing on evals.
- Build your own tiny harness in Phase 3 and reuse it in every later phase.

Evals are the actual moat. Anyone can call an API. The person who knows whether the output got better is the engineer.

### Track B — Papers & Writing
~2 papers per week regardless of phase. Three-pass method: skim structure → read figures and results → read methods only if you'll implement it.

One public write-up per phase. Blog, dev.to, or long-form README. Sebastian Raschka and Jay Alammar built careers on explainers.

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
- **GPU MODE** lecture series + Discord. Phase 2.
- **Umar Jamil (YouTube)** — from-scratch LLaMA, LoRA, DPO, Flash Attention, quantization.

Blogs: Sebastian Raschka (*Ahead of AI*), Lilian Weng, Jay Alammar, HF blog, vLLM blog, Simon Willison.

---

# PHASE 0 — Ground Floor
*Effort: 1–2 sessions. Do it today.*

Get the scaffolding up before you learn anything, so there's never friction between understanding something and recording it.

- Create the repo, write the root README with an empty progress table, commit it.
- Get GPU access working. Verify `nvidia-smi` runs. (See compute notes at the end.)
- Set up `notes/` and `benchmarks/` folders.

**Exit gate:** You can go from zero to a running notebook on a T4 in under five minutes.

---

# PHASE 1 — Foundations
### *You understand what the model actually is*
*Effort: 3–6 weeks · Syllabus themes: LLM Foundations I & II*

The goal is not "I've seen a transformer diagram." It's that the architecture holds no mystery for you.

### Unit 1.1 — Transformer architecture
*Attention Is All You Need* · GPT-1 & GPT-2 papers · Alammar's "Illustrated Transformer" · Karpathy "Let's build GPT" · Raschka Ch. 1–4 · read nanoGPT's `model.py` line by line.

### Unit 1.2 — Tokenization & pretraining objectives
Karpathy "Let's build the GPT Tokenizer" · CS336 Lecture 1 · HF Tokenizers docs · BPE, SentencePiece, tiktoken.

> Don't skip tokenization because it looks boring. Half of all weird LLM behaviour — arithmetic failures, non-English degradation, injection edge cases — traces back to the tokenizer. Especially relevant if you ever work with Indic scripts, where token efficiency is brutal.

### Unit 1.3 — Mixture of Experts
Switch Transformer · Mixtral · DeepSeekMoE / DeepSeek-V3 · HF "Mixture of Experts Explained".

### Unit 1.4 — Modern architectures & scaling laws
Raschka's "Big LLM Architecture Comparison" posts · LLaMA 1/2/3 · Qwen technical reports · Gemma · OLMo (read the *data* section — nobody else publishes it) · Kaplan 2020 · Chinchilla · *Are Emergent Abilities a Mirage?* (read this as a corrective).

### Deliverables
- `01-transformer-from-scratch/` — your own GPT-2 (~124M) in one file, plus your own BPE tokenizer. Train on TinyStories. Report loss curve and samples.
- `02-architecture-atlas/` — a table you maintain: for 8–10 open models, log norm placement, positional encoding (RoPE variants), attention type (MHA/GQA/MQA/MLA), FFN activation, dense vs MoE, vocab size, context length.
- One scaling-law experiment: train 4 tiny models at increasing size on fixed data, plot loss vs params log-log. Watching the straight line appear in *your* numbers changes how you read the field.

### 🚪 Exit gate
**Write a transformer forward pass from memory on a blank page** — shapes included, no reference. Then explain out loud, to nobody, why GQA exists and what it trades away. If you hesitate on either, you're not done.

---

# PHASE 2 — The Hardware Layer
### *You understand what the model runs on*
*Effort: 3–5 weeks · Syllabus theme: GPU Basics*

**This is the wall.** It's the hardest, least glamorous, most valuable part of the syllabus, and it's where most self-learners quietly quit and stay at the "I call APIs" level permanently. Everything in Phase 3 is incomprehensible without it and trivial with it.

### Unit 2.1 — GPU architecture
PMPP Ch. 1–6 · CS336 "GPUs, TPUs" lecture · Horace He's "Making Deep Learning Go Brrrr From First Principles" · CUDA C++ Programming Guide (skim only).

### Unit 2.2 — Parallelism
**Ultra-Scale Playbook** (primary text) · Megatron-LM · ZeRO/DeepSpeed · PyTorch FSDP docs · NCCL collectives. Understand the difference between data, tensor, pipeline, and expert parallelism well enough to say which one you'd reach for and why.

### Unit 2.3 — The physical stack
H100/H200/B200 specs — specifically memory bandwidth vs FLOPs, and note how the ratio has been getting worse · NVLink vs PCIe vs InfiniBand · Semianalysis posts on cluster design.

### Deliverables
- `03-gpu-fundamentals/` — a roofline calculator: given a model config and batch size, predict arithmetic intensity and whether you're memory- or compute-bound. Validate against real measurements.
- Three Triton kernels: vector add, fused softmax, naive matmul. Benchmark each against PyTorch and explain the gap.

### 🚪 Exit gate
**Given an arbitrary model config, batch size, and GPU, predict on paper whether the workload is memory-bound or compute-bound — then run it and be right.** Twice, on different configs.

---

# PHASE 3 — Inference & Efficiency
### *You can make it fast and small*
*Effort: 4–6 weeks · Syllabus themes: Inference, Efficient Inference & Quantization*

This is the highest-value phase for employability. Serving skills are scarce.

### Unit 3.1 — Sampling & generation
Temperature, top-k, top-p, min-p, beam search · HF `generate` source · *The Curious Case of Neural Text Degeneration*.

### Unit 3.2 — Inference math
**Kipply's "Transformer Inference Arithmetic"** (essential — read it three times) · *How to Scale Your Model* · prefill vs decode · TTFT vs ITL vs throughput · continuous batching.

> The key mental model: during decode you re-read the *entire* model's weights from HBM to produce **one** token. That's why batching gives near-free throughput and why quantization is such a large win. Internalize this and quantization, MoE serving, and edge deployment all become obvious.

### Unit 3.3 — Efficient attention & KV caching
FlashAttention 1/2/3 · PagedAttention (vLLM paper) · GQA & MQA · Multi-head Latent Attention (DeepSeek-V2) · sliding window (Mistral).

### Unit 3.4 — Quantization
MIT 6.5940 quantization lectures (best resource available) · LLM.int8() · GPTQ · AWQ · SmoothQuant · QLoRA's NF4 · FP8/FP4 · llama.cpp GGUF quant types.

### Unit 3.5 — Serving engines
vLLM docs + paper · SGLang (RadixAttention) · TensorRT-LLM · llama.cpp · Ollama · speculative decoding (Leviathan et al.) · Medusa · tensor parallelism for serving.

### Deliverables
- `04-inference-internals/` — implement KV caching yourself on your Phase 1 GPT-2. Measure tokens/sec with and without. Then write a naive continuous batcher. The gap between yours and vLLM's is the lesson.
- `05-quantization-lab/` — **the single most portfolio-valuable artifact in the plan.** Take one 8B model. Produce FP16, INT8, AWQ-4bit, and GGUF Q4_K_M. For each: VRAM, tokens/sec, and score on a small eval set. One chart, four points, real conclusions.
- Benchmark vLLM vs SGLang vs llama.cpp on identical hardware and model. Report p50/p99 under load, not averages.
- **Track A starts here:** build the reusable eval harness you'll plug into every later phase.

### 🚪 Exit gate
**Take a model that "doesn't fit" on your GPU, get it serving, and produce a defensible chart of the quality-vs-speed-vs-memory tradeoff you made.** If someone asks "why AWQ over GPTQ here?" you have an answer grounded in your own numbers.

---

# PHASE 4 — Training & Adaptation
### *You can change what the model does*
*Effort: 4–7 weeks · Syllabus theme: Fine-Tuning Fundamentals*

### Unit 4.1 — Full FT vs PEFT
LoRA · QLoRA · DoRA · PEFT survey · Raschka's LoRA-from-scratch posts · **AI Engineering Ch. 7** for the prior question: *should you fine-tune at all?* (Usually: no, try prompting and RAG first.)

### Unit 4.2 — Instruction tuning
InstructGPT · FLAN · Self-Instruct · LIMA (*less is more* — quality beats quantity) · Alpaca/Dolly formats · **chat templates**, which is where most of your bugs will actually live.

### Unit 4.3 — Preference alignment
InstructGPT (RLHF) · DPO · KTO · ORPO · SimPO · HF TRL docs · Nathan Lambert's RLHF Book.

### Unit 4.4 — RL for LLMs
PPO basics · GRPO (DeepSeek-R1 / DeepSeekMath) · RLVR (verifiable rewards) · reward models and LLM-as-judge rewards · reward hacking.

**Tooling:** Unsloth (fastest single-GPU path, runs on free Kaggle), TRL, PEFT, Axolotl or LLaMA-Factory for config-driven runs.

### Deliverable
`06-finetuning/` — a real fine-tune on a domain you care about. Full loop: dataset construction → LoRA SFT → DPO or GRPO pass → eval against base → merge → GGUF export → run locally.

**Document the failures.** The loss spike, the wrong chat template, the model that learned verbosity instead of correctness. That write-up is worth more than the model.

> Hard truth to write on a sticky note now: most fine-tuning projects fail on **data**, not method. Budget 60% of this phase on the dataset.

### 🚪 Exit gate
**Ship a fine-tuned model that measurably beats its base model on an eval set you built yourself — and be able to explain why the improvement is real and not eval contamination or a judge artifact.**

---

# PHASE 5 — Systems on Top
### *You can build things people use*
*Effort: 5–8 weeks · Syllabus themes: Reasoning, RAG, Agents, Tool Use, Agent Fine-Tuning, Evaluation*

The largest phase and the one most tutorials start with. You're arriving with foundations, which means you'll build these an order of magnitude better than the average practitioner.

### Unit 5.1 — Reasoning
Chain-of-Thought (Wei et al.) · Zero-shot CoT · Self-Consistency · Tree of Thoughts · Least-to-Most · DeepSeek-R1 (the RL-for-reasoning turning point) · o-series system cards · test-time compute scaling · **DSPy** (this is the real answer to "prompting as code").

*Deliverable:* `07-reasoning/` — on a reasoning benchmark subset, compare direct → CoT → self-consistency (n=8) → DSPy-optimized. **Plot accuracy against token cost, always.** That framing is what makes it engineering rather than prompt collecting.

### Unit 5.2 — RAG
Original RAG paper · **AI Engineering Ch. 6** · *Hands-On LLMs* Ch. 8 · chunking strategies · Anthropic's Contextual Retrieval post · hybrid search (BM25 + dense) · ColBERT/ColPali late interaction · cross-encoder rerankers · RAGAS · retrieval metrics (recall@k, MRR, nDCG) vs generation metrics.

Build bare-metal first — embedding model + FAISS/Qdrant — *then* look at LlamaIndex or Haystack, so you know what the framework hides.

*Deliverable:* `08-rag/` — RAG over a corpus you know well enough to catch it lying. Write 50 hand-made Q&A pairs **before** building the pipeline. Then ablate: naive vs semantic chunking, dense-only vs hybrid, ± reranker, ± contextual retrieval.

> The thing most people miss: RAG failures are usually *retrieval* failures, not generation failures. Instrument retrieval separately or you'll spend weeks tuning prompts to fix a chunking bug.

### Unit 5.3 — Agents & tool use
ReAct paper · **Anthropic's "Building Effective Agents"** (read it twice) · Toolformer · MCP spec at modelcontextprotocol.io · Self-RAG · CRAG · LangGraph · the honest literature on when multi-agent *hurts*.

*Deliverable:* `09-agents/` — write the agent loop from scratch first (~150 lines: LLM call → parse tool call → execute → append observation → repeat). Then rebuild with LangGraph and write up what the framework bought and what it cost. Then ship one MCP server that does something genuinely useful to you.

### Unit 5.4 — Fine-tuning for tool use
ToolLLM/ToolBench · xLAM · Gorilla · function-calling dataset formats · training on trajectories. Optional if Phase 4 exhausted you — come back to it.

### Unit 5.5 — Evaluation & observability (Track A goes formal)
**AI Engineering Ch. 3–4** · lm-evaluation-harness · HELM · benchmark contamination · LLM-as-judge failure modes (position bias, verbosity bias, self-preference) · agent benchmarks (τ-bench, AgentBench, SWE-bench, WebArena) · Langfuse (open source, self-hostable), Arize Phoenix, LangSmith · OpenTelemetry GenAI semantic conventions.

*Deliverable:* `10-eval-harness/` — mature your Phase 3 harness into something reusable. Self-host Langfuse and trace your Unit 5.2 RAG app end to end.

### 🚪 Exit gate
**A deployed system someone other than you has used, with a tracing dashboard and an eval suite that catches regressions before users do.**

---

# PHASE 6 — Specialization
### *Pick two. Skip the rest without guilt.*
*Effort: 3–5 weeks per branch · Syllabus themes: Multimodal, Edge, Security, Frontiers*

This is the branch point. Four directions, all legitimate. Depth in two beats a tour of four.

### Branch A — Multimodal
CLIP · **LLaVA** (start here, simplest architecture to grasp) · Flamingo (cross-attention approach) · Qwen-VL / Qwen-Omni reports · Whisper · vision encoders and projector layers · Unsloth VLM support.

*Deliverable:* `11-multimodal/` — fine-tune a small VLM on narrow document understanding: invoices, forms, handwritten notes, regional-language signage. **Multimodal + Indic-script documents is genuinely underserved** and makes a distinctive portfolio piece.

### Branch B — Edge
**MIT 6.5940 is the home turf here** · Phi series · Gemma small variants · Qwen 0.5B/1.5B · MobileLLM · SmolLM · distillation · structured pruning · llama.cpp internals · GGUF · ExecuTorch · MLC-LLM · ONNX Runtime · NPU/mobile inference.

*Deliverable:* `12-edge/` — run a model on the most constrained device you can reach: Android via MLC-LLM or Termux, or a Raspberry Pi. Report tokens/sec, memory, thermal throttling, battery drain. Very few people actually do this and it shows immediately.

### Branch C — Security & Privacy
**OWASP Top 10 for LLM Applications** (the canonical checklist) · Simon Willison's prompt-injection archive (the definitive ongoing record) · the *lethal trifecta*: private data + untrusted content + external communication · garak and PyRIT for red-teaming · membership inference · training-data extraction · Anthropic and OpenAI red-team publications.

*Deliverable:* `13-security/` — red-team your own Phase 5 RAG app. Attempt indirect prompt injection via a poisoned document. Document what worked, implement mitigations, re-test. Honest security write-ups are rare and hiring managers notice them.

### Branch D — Frontier architectures
Mamba & Mamba-2 (Gu & Dao) · *The Annotated S4* · RWKV · Jamba (hybrid Transformer-Mamba) · linear and hybrid attention · Qwen-Next and recent hybrid releases · diffusion language models.

*Deliverable:* pick **one** architecture, implement a minimal version, write the explainer you wish had existed when you started.

> This branch will age fastest. That's fine — the skill you're actually building is *reading a new technical report and correctly placing it*, not memorizing current SOTA.

### 🚪 Exit gate
**Two branches with real, benchmarked, documented artifacts.** Not four half-finished ones.

---

# PHASE 7 — Capstone & Signal
### *You can show it*
*Effort: 4–6 weeks*

Ship one system touching most of the stack:

> A domain-specific assistant — fine-tuned small model, quantized, served on vLLM, hybrid RAG, agentic tool use over MCP, a real eval suite, Langfuse tracing, and a documented cost/latency profile.

Then **present it**. Record a 20-minute walkthrough for YouTube, or give the talk at a meetup — Chennai has active Python and ML groups, and PyCon India takes first-time speakers. Explaining it live exposes every remaining gap, which is exactly the point.

### 🚪 Exit gate
Someone in the field watches your walkthrough and asks you a hard follow-up question — and you enjoy answering it.

---

## The minimum viable path

If life compresses and you have to cut: **Phases 1 → 2 → 3 → 4**, plus one branch from Phase 6, is a complete and defensible skill set. Phase 5 content is the most widely taught and the easiest to pick up on the job. Phases 2 and 3 are the opposite — nobody teaches them and almost nobody has them.

If you only ever finish Phase 3, you are already more useful than most people with "LLM" on their résumé.

---

## GitHub structure

Since you're documenting this — one monorepo, plus separate repos for the two or three standouts.

```
llm-engineering-journey/
├── README.md              ← front door: phase progress, gates cleared, links
├── LEARNINGS.md           ← running list of one-line insights
├── phase-0-setup/
├── phase-1-foundations/
│   ├── 01-transformer-from-scratch/
│   └── 02-architecture-atlas/
├── phase-2-hardware/
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
| 1 | Transformer from memory | ✅ | 2026-08-14 |
| 2 | Predict memory- vs compute-bound | 🔄 In progress | — |
| 3 | Serve a model that "doesn't fit" | ⬜ | — |

That table is more compelling than a commit graph, because it advertises capabilities rather than activity.

**Per-project README skeleton:**

```markdown
# <Title>
**Phase:** 3 · **Status:** Done · **Time:** ~9 hrs

## What I built
## Why it matters (the engineering problem)
## Results
<table or chart — numbers, hardware, cost, always>
## What surprised me
## What I got wrong first
## References
```

That **"what I got wrong first"** section is the differentiator. Everyone's repo shows the working version. Almost nobody shows the debugging, and that's the part that reads as real experience.

**Habits:**
- Commit small and often, not one big dump per phase.
- Push negative results. "LoRA rank 64 gave no gain over rank 16 here" is a finding.
- Every benchmark includes hardware and cost: *"A100 40GB, RunPod spot, $0.79/hr, 22 min."*
- Charts over paragraphs for results. Matplotlib is fine.
- Pin dependencies. A repo that doesn't run is a liability.

**High leverage, optional:** publish one fine-tuned model + dataset to HF Hub with a real model card · turn `notes/` into an MkDocs Material site (one afternoon) · land one small PR on vLLM, TRL, Unsloth, or llama.cpp during Phase 5+, even docs — it changes how you read codebases.

---

## Compute

- **Kaggle** — ~30 free GPU hrs/week (T4/P100). Best free option; sufficient for most of Phases 1, 3, and 4.
- **Colab** free tier for quick work; Pro if you're in Phase 4 heavily.
- **RunPod / Vast.ai** spot instances for anything bigger. A100s go for a few hundred rupees an hour.
- **Lightning AI** free monthly credits; **HF Spaces ZeroGPU** for demos.

Budget roughly ₹4,000–12,000 across the whole plan if you're disciplined. Always spot. Always kill idle machines — the most common self-study expense is a forgotten instance.

**Do not buy hardware for this.** Rent by the hour.

---

## Two last warnings

**Don't chase releases.** New models drop weekly and it will feel like falling behind. It isn't. The fundamentals in Phases 1–4 have been stable for three years; model names are noise layered on top.

**Depth beats coverage — and phases make this easy to get wrong in the other direction.** Without week boundaries, the failure mode flips from rushing to drifting. Set yourself a soft ceiling: if a phase passes eight weeks, clear the gate with whatever you have and move. Momentum is a real resource.

Start Phase 0 today, not Monday.
