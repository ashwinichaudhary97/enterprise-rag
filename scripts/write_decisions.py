"""One-off script to write docs/DECISIONS.md with correct UTF-8 encoding.

Run with: python scripts/write_decisions.py
"""

from pathlib import Path

content = """# Decision Journal

Every architectural and engineering decision, logged as it is made.

Format: date, decision, context, alternatives, rationale, reversibility.

---

## 2026-10-13 - Python 3.11 over 3.12

**Decision:** Use Python 3.11.9 (already installed) rather than upgrading to 3.12.

**Context:** Machine had 3.11.9 installed. Latest Python is 3.12/3.13. Needed to choose the target version for the venv.

**Alternatives considered:**
- Python 3.12 - newest stable
- Python 3.13 - bleeding edge
- Python 3.10 - older but stable

**Why 3.11:** ChromaDB, sentence-transformers, and torch 2.2.2 have prebuilt wheels for 3.11. Some lack 3.12/3.13 wheels, forcing source builds. 3.11 is the sweet spot for our stack.

**Reversibility:** Easy - recreate venv with different Python.

---

## 2026-10-13 - PowerShell over cmd

**Decision:** Use PowerShell as the development shell, not cmd.

**Context:** Initial terminal session was cmd. Needed to pick one for the rest of the project.

**Alternatives considered:**
- cmd - no execution policy, no setup cost
- Git Bash - Unix-style commands on Windows
- WSL - full Linux, but extra setup

**Why PowerShell:** Better Unicode handling (corpus has emojis, non-ASCII), better long-path support, aligns with modern Windows dev workflows, VS Code default. The one-time execution policy fix is worth it.

**Reversibility:** Easy - commands differ but project itself is shell-agnostic.

---

## 2026-10-13 - Pinned dependencies over floating

**Decision:** Pin every package version in requirements.txt (e.g. numpy==1.26.4 not numpy).

**Context:** Could install latest or pin to known-working versions.

**Alternatives considered:**
- Unpinned (numpy) - always latest
- Range pins (numpy>=1.26,<2) - flexible
- Exact pins - locked

**Why exact pins:** Reproducibility. A future pip install six months from now will produce the same environment. When a package breaks, we know exactly what version caused it.

**Reversibility:** Easy - edit requirements.txt.

---

## 2026-10-13 - CPU-only torch over CUDA build

**Decision:** Install torch==2.2.2 from PyTorch's CPU-only wheel index.

**Context:** Default pip install torch downloads the CUDA build (~2 GB) even when no NVIDIA GPU is present.

**Alternatives considered:**
- Default CUDA build - 2 GB waste, unused
- Older torch version - compatibility risk
- Install on Kaggle only - loses local iteration

**Why CPU-only:** Machine has no NVIDIA GPU. CPU build is 200 MB, works everywhere, and Kaggle scale-runs can use its own CUDA build separately.

**Reversibility:** Easy - pip install torch==2.2.2 (no index URL) gets the CUDA build.

---

## Template for New Entries

## YYYY-MM-DD - Short title

**Decision:** What was decided.

**Context:** Why this needed deciding.

**Alternatives considered:**
- Alt 1
- Alt 2

**Why this choice:** Rationale.

**Reversibility:** Easy / Medium / Hard.
"""

output_path = Path("docs/DECISIONS.md")
output_path.write_text(content, encoding="utf-8")
print(f"Wrote {len(content)} characters to {output_path}")