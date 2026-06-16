# Data Provenance

This repository is a cleaned portfolio edition of a Mandarin-English LLM safety evaluation project. The MIT license in `LICENSE` applies to the original source code and documentation. It does not relicense benchmark prompts, model outputs, scoring artefacts, or any upstream model- or dataset-derived material.

## Licensing Boundary

- Original source code and documentation: MIT license, see `LICENSE`.
- Benchmark prompts in `data/prompts/`: research benchmark material; not relicensed by the MIT code license.
- Captured model outputs in `data/results/`: generated evaluation artefacts; reuse may also depend on the relevant model license and acceptable-use terms.
- If a prompt or output source is unclear, treat the file as research-review material only and verify source terms before reuse or redistribution.

## Data Inventory

| Path | Role in this repository | Reuse note |
| --- | --- | --- |
| `data/prompts/1500_Chinese_prompt.jsonl` | Mandarin prompt-injection benchmark prompts used for model evaluation. | Benchmark material; not covered by the MIT code license. |
| `data/prompts/1500_English_prompt.jsonl` | Matched English prompt-injection benchmark prompts used for model evaluation. | Benchmark material; not covered by the MIT code license. |
| `data/results/glm3_results_Chinese.jsonl` | Captured ChatGLM3 Chinese responses and scoring metadata. | Generated evaluation artefact; check upstream model terms before reuse. |
| `data/results/glm3_results_English.jsonl` | Captured ChatGLM3 English responses and scoring metadata. | Generated evaluation artefact; check upstream model terms before reuse. |
| `data/results/glm4_results_Chinese.jsonl` | Captured ChatGLM4 Chinese responses and scoring metadata. | Generated evaluation artefact; check upstream model terms before reuse. |
| `data/results/glm4_results_English.jsonl` | Captured ChatGLM4 English responses and scoring metadata. | Generated evaluation artefact; check upstream model terms before reuse. |
| `data/results/llama2_results_Chinese.jsonl` | Captured LLaMA-2 Chinese responses and scoring metadata. | Generated evaluation artefact; check upstream model terms before reuse. |
| `data/results/llama2_results_Chinese_merged.jsonl` | Merged Chinese LLaMA-2 evaluation output. | Generated evaluation artefact; check upstream model terms before reuse. |
| `data/results/llama2_results_English.jsonl` | Captured LLaMA-2 English responses and scoring metadata. | Generated evaluation artefact; check upstream model terms before reuse. |

## Responsible-Use Note

The benchmark and result files contain adversarial prompts, unsafe scenarios, and model responses to those prompts. They are included only for defensive LLM security evaluation, reproducibility review, and academic portfolio inspection. They should not be treated as endorsement, operational guidance, or a prompt library for misuse.

## Reviewer Guidance

For portfolio review, inspect the benchmark structure, model-output organisation, scoring fields, and script workflow. For reuse outside review, cite the project, comply with relevant prompt/data/model terms, and keep any derived analysis clearly attributed.
