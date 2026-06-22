# LLM Safety Evaluation

[![Smoke tests](https://github.com/NAMEisNOTvailable/llm-safety-evaluation/actions/workflows/smoke.yml/badge.svg)](https://github.com/NAMEisNOTvailable/llm-safety-evaluation/actions/workflows/smoke.yml)

Mandarin-English prompt-injection evaluation project for measuring how large language models respond to adversarial instructions across languages, attack goals, and delivery styles.

This repository documents my Master of Cyber Security research work at the University of Adelaide. It focuses on reproducible LLM security evaluation with coverage-balanced prompt-injection inputs.

## Inspection Path

- **2 minutes:** read this README for the project scope, dataset layout, and reproduction requirements.
- **5 minutes:** review [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) for model-by-language refusal/compliance counts and observations.
- **10 minutes:** inspect [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md), `data/prompts/`, `data/results/`, and `scripts/validate_dataset.py` to check data boundaries and validation coverage.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Research focus | Cross-lingual prompt-injection behaviour in Mandarin and English |
| Dataset | 1,500 matched Mandarin-English prompt pairs |
| Models evaluated | ChatGLM3-6B, ChatGLM4-9B, and LLaMA-2-13B |
| Evaluation labels | Human-reviewed final labels: Complete Refusal, Partial Compliance, Full Compliance |
| Main output | Preserved model responses, Gatekeeper v2 automatic labels, and human-reviewed final labels |

## What This Demonstrates

- Designed a coverage-balanced benchmark instead of relying on ad hoc attack prompts.
- Compared refusal and compliance behaviour across Chinese, English, and bilingual attack styles.
- Built Python scripts for repeatable model inference and response capture.
- Organised model outputs so results can be reviewed, audited, and compared by language/model.
- Preserved original model answers while separating automatic scoring from final human review.
- Framed results as security evaluation evidence, with attention to partial compliance and risk interpretation.

## Repository Structure

```text
data/
  prompts/      Matched Mandarin-English benchmark prompt files
  results/      Captured model output JSONL files with automatic and human labels
docs/           Result summary and project notes
scripts/        Model-running and evaluation scripts
README.md       Project overview and inspection guide
```

## Data Layout

| Path | Purpose |
| --- | --- |
| `data/prompts/1500_Chinese_prompt.jsonl` | Mandarin prompt-injection benchmark set |
| `data/prompts/1500_English_prompt.jsonl` | English matched benchmark set |
| `data/results/*_results_Chinese.jsonl` | Chinese model responses with Gatekeeper v2 labels and human-reviewed final labels |
| `data/results/*_results_English.jsonl` | English model responses with Gatekeeper v2 labels and human-reviewed final labels |

Result files keep the original `model_answer` once. The earlier automatic label is stored as `evaluation_result_gatekeeper_v2`; the final human-reviewed label is stored as `evaluation_result`; `adjudication_reason` records the review rationale.

## Script Entry Points

| Script | Purpose |
| --- | --- |
| `scripts/model_runner_evaluation.py` | General runner/evaluation workflow used for GLM3-style experiments |
| `scripts/prompt_glm4_Chinese.py` | ChatGLM4 Chinese prompt evaluation |
| `scripts/prompt_glm4_English.py` | ChatGLM4 English prompt evaluation |
| `scripts/prompt_llama2_Chinese.py` | LLaMA-2 Chinese prompt evaluation |

## Environment

Recommended runtime: Python 3.10 with PyTorch, Transformers, Accelerate, Safetensors, SentencePiece, Protobuf, Einops, and TQDM. Install the base environment with:

```bash
pip install -r requirements.txt
```

Full model inference requires externally downloaded model weights and a CUDA-capable PyTorch installation sized for the selected model. Optional acceleration packages such as `bitsandbytes` or `flash-attn` can be installed separately when supported by the local GPU/Linux environment.

Run scripts from the repository root so default relative paths resolve correctly:

```bash
python scripts/model_runner_evaluation.py
```

Model checkpoints and offload storage are configurable across experiment machines. Use `--model` for a Hugging Face model id or local checkpoint path, and `--offload-dir` for Hugging Face/Accelerate offload files:

```bash
python scripts/prompt_llama2_Chinese.py \
  --model meta-llama/Llama-2-13b-chat-hf \
  --offload-dir .cache/hf_offload
```

The same values can be supplied through `LLAMA2_MODEL` and `HF_OFFLOAD_DIR` when running repeated experiments.

## Reproduction Commands

The checked-in JSONL outputs were initially produced from the scripts below. Full model inference requires externally downloaded model weights, enough GPU memory for the selected model, and any model-specific access approval or local checkpoint mirror required by the model provider.

The current result files also include a later human review pass. The original model answers remain in `model_answer`, the earlier Gatekeeper v2 automatic labels remain in `evaluation_result_gatekeeper_v2`, and the human-reviewed final labels used in `docs/RESULTS_SUMMARY.md` are stored in `evaluation_result`.

```bash
# ChatGLM3-6B on English prompts
python scripts/model_runner_evaluation.py \
  --in data/prompts/1500_English_prompt.jsonl \
  --out data/results/glm3_results_English.jsonl \
  --model THUDM/chatglm3-6b

# ChatGLM3-6B on Mandarin prompts
python scripts/model_runner_evaluation.py \
  --in data/prompts/1500_Chinese_prompt.jsonl \
  --out data/results/glm3_results_Chinese.jsonl \
  --model THUDM/chatglm3-6b \
  --zh-mode han_en \
  --no-en-only-hard

# ChatGLM4-9B on English prompts
python scripts/prompt_glm4_English.py \
  --in data/prompts/1500_English_prompt.jsonl \
  --out data/results/glm4_results_English.jsonl \
  --model ZhipuAI/glm-4-9b-chat

# ChatGLM4-9B on Mandarin prompts
python scripts/prompt_glm4_Chinese.py \
  --in data/prompts/1500_Chinese_prompt.jsonl \
  --out data/results/glm4_results_Chinese.jsonl \
  --model ZhipuAI/glm-4-9b-chat \
  --zh-mode han_en \
  --no-en-only-hard

# LLaMA-2-13B on English prompts
python scripts/model_runner_evaluation.py \
  --in data/prompts/1500_English_prompt.jsonl \
  --out data/results/llama2_results_English.jsonl \
  --model meta-llama/Llama-2-13b-chat-hf \
  --offload-dir .cache/hf_offload

# LLaMA-2-13B on Mandarin prompts
python scripts/prompt_llama2_Chinese.py \
  --in data/prompts/1500_Chinese_prompt.jsonl \
  --out data/results/llama2_results_Chinese.jsonl \
  --model meta-llama/Llama-2-13b-chat-hf \
  --offload-dir .cache/hf_offload \
  --zh-mode han_en \
  --no-en-only-hard
```

To validate the checked-in artefacts without loading model weights, run:

```bash
python scripts/validate_dataset.py
```

## Research Questions

- Do Mandarin prompt-injection variants produce different refusal/compliance patterns from English variants?
- Which attack goals and delivery modalities expose higher-risk model behaviour?
- How can LLM security experiments be made reproducible and coverage-balanced?
- How should partial compliance be separated from full harmful compliance?

## Related Project

This repository complements [`llm-defend`](https://github.com/NAMEisNOTvailable/llm-defend), which contains a larger Chinese prompt-injection dataset composer and deterministic deduplication pipeline.

## License and Data

Original source code and documentation are licensed under the MIT License. Benchmark prompts, model outputs, scoring artefacts, and other data files are not relicensed by this repository. See [Data Provenance](DATA_PROVENANCE.md) before reusing prompts, results, or model-output files.

## Status

Academic research project organised for review of benchmark files, model outputs, and the evaluation workflow.
