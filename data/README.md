# Data Directory

This directory keeps benchmark prompts separate from generated model outputs and review labels.

```text
prompts/   Matched Mandarin-English prompt-injection benchmark files
results/   Captured model responses with automatic and human-reviewed labels
```

The files are kept in JSONL format so each prompt or response can be inspected, filtered, and scored independently. Result files preserve the original `model_answer`, retain the earlier Gatekeeper v2 automatic label in `evaluation_result_gatekeeper_v2`, and store the human-reviewed final label in `evaluation_result`. The review rationale is stored in `adjudication_reason`.

Result files do not duplicate prompt text. Join prompts and results by `idx` when prompt text is needed for analysis.

Local reruns should write raw model outputs to `../build/raw_results/` first. The checked-in `results/` files should be regenerated through `../scripts/merge_human_review_labels.py` after human review, rather than overwritten directly by inference scripts.

These prompt and result files are research artefacts and are not covered by the repository's MIT code license. See [`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) for source, model-output, and reuse notes.
