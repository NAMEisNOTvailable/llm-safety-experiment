# Judge accuracy evaluation

This branch adds `scripts/evaluate_pilot_judge.py`, which evaluates four channels stored in the checked-in result files:

- final fused `evaluation_result`;
- deterministic `meta.label_rules`;
- first `meta.label_llm` judgment;
- second/rejudge label when present.

The current embedded pilot contains 37 fact-pollution reviews. Eight `Invalid/Unclear` records are excluded from the three-way denominator, leaving 29 comparable responses across 15 prompt IDs.

The script reports coverage, covered and strict accuracy, macro-F1, balanced accuracy, Cohen's kappa, per-label metrics, confusion matrices, a majority-class baseline, and cluster-bootstrap accuracy intervals grouped by prompt `idx`.

Run locally:

```bash
python scripts/evaluate_pilot_judge.py
```

Outputs are written to `judge-evaluation-output/`.

The pilot is a targeted single-reviewer regression set, not an unbiased human-gold estimate over all repository outputs. A publication-quality accuracy estimate requires a pre-specified stratified sample, two independent bilingual annotators, and adjudication.