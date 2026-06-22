# Scripts Directory

This directory contains the model-running and evaluation scripts used for the LLM safety experiment.

Run scripts from the repository root so default paths such as `data/prompts/...` and `build/raw_results/...` resolve correctly.

```bash
python scripts/model_runner_evaluation.py
```

Model-running scripts write raw rerun outputs under `build/raw_results/` by default. That directory is ignored by Git so local reruns do not overwrite the human-reviewed files in `data/results/`.

Use `merge_human_review_labels.py` to regenerate final checked-in results after a human review pass:

```bash
python scripts/merge_human_review_labels.py \
  --raw-dir build/raw_results \
  --manual-dir ../outputs/final_classification_json_20260622 \
  --out-dir data/results \
  --label-version 2026-06-22
```

Model paths and offload storage are configurable:

- `--model` accepts either a Hugging Face model id or a local checkpoint path.
- `--offload-dir` controls where Hugging Face/Accelerate writes offloaded checkpoint shards.
- `LLAMA2_MODEL` can override the default LLaMA-2 model id for `prompt_llama2_Chinese.py`.
- `HF_OFFLOAD_DIR` can override the default `.cache/hf_offload` directory.

The scripts are preserved as research artefacts for inspecting the inference and scoring workflow.
