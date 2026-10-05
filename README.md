# RL-AR: Reinforcement Learning-Based Adaptive Retrieval for Efficient RAG

Code and artifacts for the IEEE Access manuscript (under review). RL-AR selects how many passages to retrieve
per query, formulated as a cost-aware contextual bandit and optimized with single-step PPO. RL-AR-Lite decides
from retrieval-score and query features before any LLM call.

## Repository layout

~~~
notebooks/
  rl_ar_unified.ipynb        # full pipeline for one (model, dataset) config; edit the CONFIG cell only
  02_summary.ipynb           # aggregates all 10 configs into the paper tables and figures
  03_revision_followup.ipynb # lambda-sweep frontier, matched budgets, transfer, decision profile
data/
  pins.json                  # pinned Hugging Face revisions of every model and dataset
  {nq,squad}/splits.json     # exact train/val/test question IDs (article-grouped, seed 2026)
  {nq,squad}/corpus.jsonl.gz # retrieval corpus (102,332 NQ / 31,094 SQuAD passages)
runs/{dataset}/{model}/
  gen/*.jsonl.gz             # cached greedy generations for k = 0..3 and FLARE
  tables/                    # per-config result tables (CSV/LaTeX)
  results/*.pkl              # per-query, per-seed policy results
  latency_test.jsonl.gz      # per-stage latency at batch size 1
  manifest.json              # versions, checkpoints, hyperparameters, compute
summary/                     # cross-config tables and figures used in the paper
~~~

## Reproducing

1. Run `notebooks/rl_ar_unified.ipynb` on a 16 GB GPU (we used a Tesla T4) once per config:
   `MODEL_KEY` in {llama1b, llama3b, qwen3b, mistral7b, llama8b}, `DATASET_KEY` in {nq, squad}.
   Llama checkpoints are gated: set an `HF_TOKEN` with access.
   To reuse our cached generations instead of regenerating, copy `data/` and `runs/` into `OUT_ROOT`
   (decompress the `.gz` files first) and the notebook skips every finished stage.

Corpus embeddings (`corpus_emb.npy`) are not included because they exceed GitHub's file size limit;
the notebook rebuilds them deterministically in a few minutes.

## Citation

See `CITATION.cff`.

## License

MIT
