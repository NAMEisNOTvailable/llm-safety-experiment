# LLM Safety Evaluation

Mandarin-English prompt-injection evaluation project for measuring how large language models respond to adversarial instructions across languages, attack goals, and delivery styles.

This repository is a cleaned portfolio version of my Master of Cyber Security research work at the University of Adelaide. It focuses on reproducible LLM security evaluation rather than collecting isolated jailbreak examples.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Research focus | Cross-lingual prompt-injection behaviour in Mandarin and English |
| Dataset | 1,500 matched Mandarin-English prompt pairs |
| Models evaluated | ChatGLM3-6B, ChatGLM4-9B, and LLaMA-2-13B |
| Evaluation labels | Complete Refusal, Partial Compliance, Full Compliance |
| Main output | Model response files and scoring workflow for comparative safety analysis |

## What This Demonstrates

- Designed a coverage-balanced benchmark instead of relying on ad hoc attack prompts.
- Compared refusal and compliance behaviour across Chinese, English, and bilingual attack styles.
- Built Python scripts for repeatable model inference and response capture.
- Organised model outputs so results can be reviewed, audited, and compared by language/model.
- Framed results as security evaluation evidence, with attention to partial compliance and risk interpretation.

## Repository Structure

```text
data/
  prompts/      Matched Mandarin-English benchmark prompt files
  results/      Captured model output JSONL files
scripts/        Model-running and evaluation scripts
README.md       Project overview and reviewer guide
```

## Data Layout

| Path | Purpose |
| --- | --- |
| `data/prompts/1500_Chinese_prompt.jsonl` | Mandarin prompt-injection benchmark set |
| `data/prompts/1500_English_prompt.jsonl` | English matched benchmark set |
| `data/results/*_results_Chinese.jsonl` | Chinese model responses |
| `data/results/*_results_English.jsonl` | English model responses |

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

Academic research portfolio project. The repository is organised for reviewers who want to inspect the benchmark files, model outputs, and evaluation workflow quickly.
