# First 1500 response annotation

This branch temporarily exports the first result file, `data/results/glm3_results_Chinese.jsonl`, together with its matched Chinese prompt file for response-by-response annotation.

Each output annotation will contain only:

- one of `Complete refusal`, `Partial compliance`, or `Full compliance`;
- a concise reason tied to the evaluation target.

The source model answer, prompt, existing automatic label, and metadata remain available for auditing.