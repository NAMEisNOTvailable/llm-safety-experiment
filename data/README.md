# Data Directory

This directory keeps benchmark prompts separate from generated model outputs.

```text
prompts/   Matched Mandarin-English prompt-injection benchmark files
results/   Captured model responses used for safety/compliance analysis
```

The files are kept in JSONL format so each prompt or response can be inspected, filtered, and scored independently.

These prompt and result files are research artefacts and are not covered by the repository's MIT code license. See [`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md) for source, model-output, and reuse notes.
