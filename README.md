# Language Models Trained With Helpers Stop Learning, and Forget, What the Helpers Know

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24124317.svg)](https://doi.org/10.5281/zenodo.24124317)

Code, run commands and training logs for the paper *Language Models Trained With Helpers Stop Learning, and Forget, What the
Helpers Know: Kamin Blocking and a One-Line Fix* (Jungyun Han). **Paper: [paper.pdf](paper.pdf)**

When a language model is trained together with a helper whose predictions enter its likelihood (a copy head, a cache, a frozen
expert, a tool, or a retrieved passage), it stops learning, and even forgets, what the helper supplies. The mixture gradient is the
shared prediction error of the Rescorla–Wagner rule (Kamin blocking); adding the model's own cross-entropy (`--lmaux 1.0`) removes it.

- `code/` — training (`train.py`), synthetic tasks (`synth_*.py`), analysis scripts and every run script (`round*.sh`).
  `code/README.md` maps each table and figure of the paper to the command that produced it.
- `logs/` — training logs and analysis outputs; every number in the paper is computed from these or from released checkpoints.

Data: FineWeb-edu (GPT-2 BPE), prepared with `code/prep.py`. Released Copy-Paste LMs are loaded from the Hugging Face hub.
