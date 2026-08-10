# LLM Engineering
 
Working through an engineering-focused LLM curriculum, learned using implementation.
Each phase ends with an **exit gate** — a concrete thing I must be able to build or explain before moving on.
 
**Hardware:** MacBook Pro M4, 16GB unified (120 GB/s) · Kaggle free tier (T4 ×2 / P100) · occasional rented GPU.
Details and quota log in [HARDWARE.md](HARDWARE.md). Running insights in [LEARNINGS.md](LEARNINGS.md).
 
## Progress
 
| Phase | Exit gate | Status | Cleared |
|---|---|---|---|
| 0 · Setting Up | Kaggle GPU + and local 8B model with tok/s | C | Y |
| 1 · Foundations | Transformer forward pass from memory, shapes included | N | — |
| 2 · Hardware Layer | Predict memory- vs compute-bound, then be right — twice | N | — |
| 3 · Inference | Serve a model that "doesn't fit"; defend the tradeoff chart | N | — |
| 4 · Adaptation | Fine-tune that beats base on my own eval set | N | — |
| 5 · Systems | Deployed system with tracing + regression-catching evals | N | — |
| 6 · Specialization | Two branches, benchmarked and documented | N | — |
| 7 · Capstone | Walkthrough published; hard questions welcome | N | — |
 
Status key: N: not started - P: in progress - C: cleared
 
## Projects
 
| Phase | Project | Notes |
|---|---|---|
| 2 | [roofline-across-three-chips](phase-2-hardware/roofline-across-three-chips/) | M4 vs T4 vs P100 — memory-bound vs compute-bound |
| 3 | [quantization-lab](phase-3-inference/05-quantization-lab/) | Quantization × hardware matrix |

 
## Conventions
 
- Every benchmark number is labelled with the chip it ran on. `18 tok/s` alone is noise;
  `18 tok/s (M4, Q4_K_M, 120 GB/s)` is a result.
- Negative results get pushed too. "No gain from rank 64 over rank 16" is a finding.
- Every project README includes a **What I got wrong first** section.