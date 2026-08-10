# Kaggle bootstrap
 
Paste these as the first cells of any new notebook. Notebook settings first:
**Accelerator → GPU (T4 ×2 or P100)** and **Internet → On**.
 
## Cell 1 — what did I actually get?
 
```python
!nvidia-smi
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[{i}] {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
print("bf16 supported:", torch.cuda.is_bf16_supported())
```
 
`sm_75` = T4 (no bf16, no FlashAttention 2). `sm_60` = P100 (no tensor cores).
 
## Cell 2 — clone the repo
 
```python
from kaggle_secrets import UserSecretsClient
import os, subprocess
 
TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
USER  = "YOUR_GITHUB_USERNAME"
REPO  = "llm-engineering-journey"
 
os.chdir("/kaggle/working")
if not os.path.exists(REPO):
    subprocess.run(
        ["git", "clone", f"https://{USER}:{TOKEN}@github.com/{USER}/{REPO}.git"],
        check=True,
    )
os.chdir(REPO)
subprocess.run(["git", "config", "user.email", "you@example.com"], check=True)
subprocess.run(["git", "config", "user.name", USER], check=True)
print("cwd:", os.getcwd())
```
 
Set the secret once: **Add-ons → Secrets → Add secret**, label `GITHUB_TOKEN`,
value = a GitHub fine-grained PAT with Contents: read/write on this repo.
 
## Cell 3 — push results back
 
```python
!git add -A && git commit -m "kaggle: <what ran>" && git push
```
 
## Quota hygiene
 
- Debug on the Mac at tiny scale first. Arrive at Kaggle with working code.
- Long jobs: **Save Version → Run All** so it executes in the background.
- Kill idle sessions manually — idle burns quota.
- Checkpoint and push every epoch. Session state does not survive.
