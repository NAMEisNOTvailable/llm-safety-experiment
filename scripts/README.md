# Scripts Directory

This directory contains the model-running and evaluation scripts used for the LLM safety experiment.

Run scripts from the repository root so default paths such as `data/prompts/...` and `data/results/...` resolve correctly.

```bash
python scripts/model_runner_evaluation.py
```

Model paths and offload storage are configurable:

- `--model` accepts either a Hugging Face model id or a local checkpoint path.
- `--offload-dir` controls where Hugging Face/Accelerate writes offloaded checkpoint shards.
- `LLAMA2_MODEL` can override the default LLaMA-2 model id for `prompt_llama2_Chinese.py`.
- `HF_OFFLOAD_DIR` can override the default `.cache/hf_offload` directory.

The scripts are preserved as research artefacts for inspecting the inference and scoring workflow.
