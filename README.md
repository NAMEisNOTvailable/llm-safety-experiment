# LLM Safety Experiment

Evaluation scripts and experiment notes for Mandarin-English prompt-injection security research.

This repository supports my Master of Cyber Security thesis work at the University of Adelaide: **A Coverage-Balanced Chinese Prompt-Injection Generator and Pipeline**. The project focuses on cross-lingual LLM security, Chinese prompt-injection behaviour, and reproducible evaluation design.

## Portfolio Summary

- Built a Mandarin-English prompt-injection benchmark with **1,500 matched prompt pairs**.
- Used a **5 x 5 goal-by-modality taxonomy** to compare model behaviour across attack goals and delivery styles.
- Evaluated **ChatGLM3-6B**, **ChatGLM4-9B**, and **LLaMA-2-13B** through a unified inference and scoring workflow.
- Applied a Gatekeeper-style evaluation scheme with **Complete Refusal**, **Partial Compliance**, and **Full Compliance** labels.
- Designed the work around reproducibility, coverage balance, clear metrics, and risk-aware interpretation rather than ad hoc prompt examples.

## Research Focus

The project investigates how Chinese and bilingual prompt-injection patterns affect model compliance and refusal behaviour. It is intended as a security evaluation workflow rather than a jailbreak collection.

Key questions:

- Do Mandarin and bilingual prompt-injection variants produce different refusal/compliance patterns?
- Which attack goals and delivery modalities expose higher-risk behaviour?
- How can prompt-injection experiments be made more reproducible and coverage-balanced?
- How should evaluation labels distinguish partial compliance from full harmful compliance?

## Skills Demonstrated

- AI safety and LLM security evaluation
- Prompt-injection taxonomy design
- Multilingual robustness analysis
- Python-based experiment pipelines
- Security-oriented model evaluation
- Data cleaning, scoring, and reproducibility checks
- Technical writing and research communication

## Related Work

This repository complements [`llm-defend`](https://github.com/NAMEisNOTvailable/llm-defend), which contains a larger Chinese prompt-injection dataset composer and deterministic deduplication pipeline.

## Status

Research portfolio project. The README is written for recruiters and reviewers who want a quick view of the project purpose, methods, and security relevance.