# How KV-Caches Make LLM Inference Fast

**Introduction to MLOps · lecture slides and hands-on notebook**
Mahdi Abbasi · Probationary lecture, AI Infrastructure · THWS Würzburg · October 2026

This repository teaches one core idea of LLM inference: **why generating text gets slower with every token, and how the KV cache fixes it.**
Everything runs on a laptop CPU with **GPT-2 small** (124M parameters) and the prompt *"Machine learning can"*.

---

## Contents

| File | What it is |
|---|---|
| [`KV-Cache.ipynb`](KV-Cache.ipynb) | Companion notebook. The code shown on the slides, with real outputs and timing plots. |
| [`KV-Cache Lecture.pdf`](KV-Cache%20Lecture.pdf) | Lecture slides (about 20 minutes, plus backup slides). |
| [`requirements.txt`](requirements.txt) | Python packages. |

---

## What you will learn

1. **Text generation:** how GPT-2 turns a prompt into *one* next token (tokenizer → 12 blocks → LM head → argmax).
2. **Autoregressive generation:** each new token goes back into the input, so a simple loop re-reads the whole sequence at every step.
3. **Why it gets slow:** the time per token **rises** (about 40 → 150 ms over 50 tokens), because the keys and values of old tokens are computed again and again.
4. **Q, K, V and the causal mask:** a new token only needs its own query, but it needs the keys and values of *all* earlier tokens.
5. **The KV cache:** old keys and values never change, so we compute them once, store them, and reuse them.
6. **Prefill and decode:** prefill reads the whole prompt (→ *TTFT*, time to first token); decode makes one token per step (→ *TPOT*, time per output token).
7. **The price:** the cache needs memory (2 × layers × hidden × bytes per token).

---

## Quick start

```bash
git clone https://github.com/Mahdi-abbasi1358/KV-Cache.git
cd KV-Cache

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch transformers matplotlib numpy jupyterlab
jupyter lab KV-Cache.ipynb
```

> **Note on `requirements.txt`:** it is a full export of the author's Windows environment
> (it also contains TensorFlow, Keras and the Windows-only package `pywinpty`).
> On Linux or macOS, `pip install -r requirements.txt` may fail; use the short install line above instead.
> The notebook only needs `torch`, `transformers`, `matplotlib` and `numpy`.

The first run downloads GPT-2 (about 500 MB) from Hugging Face. No GPU is needed.

---

## Notebook structure

| Part | Topic | What you do |
|---|---|---|
| 1 | Load GPT-2 | Load the model and tokenizer; find why `c_attn` has 2304 = 3 × 768 outputs (Q, K, V). |
| 2 | One token by hand | Tokenize the prompt, run one forward pass, take the **last position**, pick the best token → `" be"`. |
| 3 | Autoregressive generation | Feed each new token back into the input. |
| 4 | The simple loop (no cache) | Generate 50 tokens with `generate_token` and measure the time of every step. |
| 5 | Attention and the causal mask | Look at
