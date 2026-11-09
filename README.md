# RL-AR: Cost-Aware Reinforcement Learning for Adaptive Retrieval Depth in Retrieval-Augmented Generation

Code, notebooks with outputs, cached generations and result files for the paper.

## Contents
- `notebooks/rl_ar_pipeline.ipynb`: full pipeline (data, retrieval, response caching, latency profiling, policies, baselines, ablations, Self-RAG). Set `MODEL_KEY` and `DATASET_KEY` in the config cell.
- `notebooks/runs/<dataset>_<model>.ipynb`: the executed pipeline for each of the 15 settings, with outputs.
- `notebooks/cross_dataset_transfer.ipynb`: frozen-policy transfer between NQ and SQuAD.
- `notebooks/confidence_probe_diagnostic.ipynb`: prefill confidence-probe diagnostic (Llama-3.2-1B).
- `data/<dataset>/`: sampled splits (question identifiers), retrieval corpus, data manifest; `data/pins.json`: pinned model and dataset revisions.
- `runs/<dataset>/<model>/`: cached generations for every depth (`gen/`), feature and reward tables, latency profiles, trained-policy results (`results/`), per-run tables and figures.
- `results/cross_dataset_transfer/`: frozen-policy transfer results.
- `results/confidence_probe/`: tables of the confidence-probe diagnostic.

## Setup
Models: Llama-3.2-1B/3B-Instruct, Qwen2.5-3B-Instruct, Mistral-7B-Instruct-v0.3 (NF4), Llama-3.1-8B-Instruct (NF4). Retriever: BGE-small-en-v1.5 with an exact FAISS index. Datasets: NaturalQuestions (DPR dev), SQuAD v1.1, HotpotQA (distractor dev). Hardware: one NVIDIA T4 (16 GB).

`pip install -r requirements.txt`. Gated models need a Hugging Face token in the environment variable `HF_TOKEN`.

Policies train on the CPU from the cached tables in `runs/`, so every table in the paper can be reproduced without a GPU.
