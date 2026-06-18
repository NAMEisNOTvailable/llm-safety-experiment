#!/usr/bin/env python3
import json
from pathlib import Path
from scripts import probe_nli as probe

out_dir = Path("build/glm3_chinese_reasoned")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "nli_probe.jsonl"
with out_path.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps({"model": probe.MODEL, "id2label": probe.id2label, "entailment_id": probe.ent_id}, ensure_ascii=False) + "\n")
    for idx in probe.IDS:
        handle.write(json.dumps({"idx": idx, **probe.by[idx]}, ensure_ascii=False) + "\n")
print(f"Wrote {out_path}")
