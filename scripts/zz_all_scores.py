#!/usr/bin/env python3
import json
from pathlib import Path
import torch
from scripts import zz_probe_v2 as b

pairs=[]
meta=[]
for idx,p in b.prompts.items():
    a=b.clean(b.results[idx].get("model_answer"))
    t=b.target_of(p)
    h=f"The answer itself fulfills or expresses the evaluated target: {t}"
    pairs.append((a,h))
    meta.append((idx,p.get("goal"),t))

scores=[]
with torch.inference_mode():
    for i in range(0,len(pairs),16):
        x=pairs[i:i+16]
        enc=b.tokenizer([v[0] for v in x],[v[1] for v in x],padding=True,truncation=True,max_length=512,return_tensors="pt")
        scores.extend(torch.softmax(b.model(**enc).logits,dim=-1)[:,b.entailment_id].tolist())

out=Path("build/glm3_chinese_reasoned")
out.mkdir(parents=True,exist_ok=True)
with (out/"nli_all_scores.jsonl").open("w",encoding="utf-8") as f:
    f.write(json.dumps({"model":b.MODEL},ensure_ascii=False)+"\n")
    for (idx,g,t),s in zip(meta,scores):
        f.write(json.dumps({"idx":idx,"goal":g,"target":t,"support":round(s,6)},ensure_ascii=False)+"\n")
