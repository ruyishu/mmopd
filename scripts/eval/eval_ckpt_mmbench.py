"""MMBench-EN dev eval of a student ckpt (HF generate, batched).
python scripts/eval/eval_ckpt_mmbench.py CONFIG CKPT_DIR OUT_DIR [N_QUESTIONS=0 -> all 1164]
All dataset rows (incl. circular rotations) are asked; reports single-pass (original rows) + circular.
Sampling as base eval (temp .7, top_p .8, top_k 20; HF has no presence_penalty), max 64 tokens, seed 0."""
import io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from PIL import Image
from mmb_common import load_dev, build_prompt, extract, score
from mm_opd.imgdata import VisionPrep
from mm_opd.model.student import Student
from mm_opd.train.common import load_cfg, seed_all
cfg_p, ck, out = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 0
BS = int(os.environ.get("EVAL_BS", "32"))
cfg = load_cfg(cfg_p); seed_all(0); os.makedirs(out, exist_ok=True)
dev = cfg["student"]["device"]
st = Student(cfg, dev); st.projector.load_state_dict(torch.load(f"{ck}/projector.pt"))
if os.path.isdir(f"{ck}/lora"):
    from peft import PeftModel
    st.llm = PeftModel.from_pretrained(st.llm, f"{ck}/lora").merge_and_unload()
st.llm.eval(); prep = VisionPrep(cfg); tok = st.tok; tok.padding_side = "left"
df = load_dev(N or None)
recs, t0 = [], time.time()
for i in range(0, len(df), BS):
    b = df.iloc[i:i + BS]; qs, ims, ol = [], [], []
    for _, r in b.iterrows():
        p, opts = build_prompt(r); qs.append(p); ol.append(opts)
        im = r["image"]; ims.append(Image.open(io.BytesIO(im["bytes"] if isinstance(im, dict) else im)).convert("RGB"))
    enc = tok([st.prompt_text(q) for q in qs], add_special_tokens=False, return_tensors="pt", padding=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        emb = st.embed(enc["input_ids"].to(dev), prep(ims))
        g = st.llm.generate(inputs_embeds=emb, attention_mask=enc["attention_mask"].to(dev), do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=64, pad_token_id=tok.pad_token_id)
    for (_, r), seq, opts in zip(b.iterrows(), g, ol):
        txt = tok.decode(seq, skip_special_tokens=True).strip(); pred, method = extract(txt, opts)
        recs.append({"index": int(r["index"]), "gold": r["answer"], "pred": pred, "method": method, "correct": pred == str(r["answer"]).strip(), "output": txt})
    if (i // BS) % 20 == 0: print("progress", len(recs), "/", len(df), round(time.time() - t0), "s", flush=True)
with open(f"{out}/mmbench_outputs.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
res = score(recs); res.update({"ckpt": ck, "seconds": round(time.time() - t0, 1), "student_extractor": "letter_start/letter_phrase/option_text; unparseable=wrong"})
json.dump(res, open(f"{out}/mmbench_result.json", "w"), indent=2); print("RESULT", json.dumps(res))
