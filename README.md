Phi-3-mini weekend probe
========================

Tiny demo to:
- Grab Phi-3-mini hidden states for three prompt groups (refusal, jailbreak, alignment).
- StreamMem-style top-k token compression vs full mean-pool.
- Train probes for static accuracy and a short A→B→C cyclic schedule.

Quick setup (run later)
-----------------------
- `./setup_env.sh` (creates `.venv`, installs `requirements.txt`).
- Model: `microsoft/Phi-3-mini-4k-instruct` (~7 GB fp16 from HF). Set `HF_TOKEN` if needed.

Main commands (run inside venv)
-------------------------------
- Collect: `python phi3_pipeline.py collect --model-id microsoft/Phi-3-mini-4k-instruct --top-k 8 --max-new-tokens 128`
- Static probe: `python phi3_pipeline.py static-probe --features outputs/features.npz`
- Cyclic probe: `python phi3_pipeline.py cyclic-probe --features outputs/features.npz --cycles 3`

Files
-----
- `phi3_pipeline.py` — collection + probes.
- `requirements.txt` — deps.
- `setup_env.sh` — venv helper.
- `outputs/` — created when you run collect.
