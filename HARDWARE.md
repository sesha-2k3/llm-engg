# Hardware & Compute
 
## Local — MacBook Pro M4
 
| | |
|---|---|
| Memory | 16GB unified |
| Memory bandwidth | 120 GB/s |
| GPU compute | Metal / MPS · no CUDA, ever |
| Default GPU memory ceiling | ~10.6GB (macOS reserves the rest) |
 
**Raise the Metal wired-memory limit** (resets on reboot):
 
```bash
sudo sysctl iogpu.wired_limit_mb=12288
```
 
Don't push this much higher — starving macOS causes swapping and instability.
 
**What fits locally:** 8B at 4-bit is comfortable · 14B at 4-bit works with other apps closed · 32B is out.
 
## Kaggle free tier
 
| | |
|---|---|
| Quota | ~30 GPU-hours/week, resets weekly |
| Options | 1× P100 (16GB, ~732 GB/s, no tensor cores) · 2× T4 (16GB each, ~320 GB/s, FP16 tensor cores) |
| Session cap | up to 12 hours |
| Persistent storage | 20GB |
 
Pick the accelerator deliberately: **P100 for bandwidth-bound work, T4×2 for anything training or multi-GPU.**
 
### Known gaps
 
- **FlashAttention 2** needs sm_80+; T4 is sm_75. Won't run. → rent once.
- **No bf16** on Turing/Pascal. fp16 only; expect occasional loss instability.
- **Nsight Compute blocked** — no GPU counter permissions. Use `torch.profiler` and CUDA events. → rent for real profiling.
- **No FP8 / Hopper features.**
- vLLM benchmarks run but aren't representative of modern serving hardware.
 
### Compiling CUDA in a notebook
 
```python
!pip install nvcc4jupyter
%load_ext nvcc4jupyter
# then %%cuda at the top of a cell
```
 
Arch flags: `sm_75` (T4), `sm_60` (P100).
 
## Rented, for the gaps
 
Vast.ai / RunPod spot, 4090 or L4, with root so Nsight works. Budget ~two 6-hour sessions
across the whole plan. Modal is a good alternative — write Python locally, execute on cloud GPU.
 
## Quota log
 
| Week | Kaggle hrs used | Rented | Spent | What it bought |
|---|---|---|---|---|
| | | | | |