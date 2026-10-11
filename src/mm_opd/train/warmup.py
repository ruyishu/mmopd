"""Stage 1: frozen LLM (+ frozen ViT), train projector only on COCO captions.
Loss = next-token CE on caption tokens (+ <|im_end|>).
python -m mm_opd.train.warmup --config configs/demo_1p7b.yaml [--max_steps N --micro M --accum A --limit K --out DIR]
"""
import argparse, math, os, time
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from mm_opd.imgdata import JsonlImages, VisionPrep
from mm_opd.model.student import Student
from mm_opd.train.common import load_cfg, log, peak_mem, save_projector, seed_all


def run_warmup(config_path, max_steps=None, micro=None, accum=None, limit=None, out=None):
    cfg = load_cfg(config_path); W = cfg["warmup"]
    seed_all(cfg["seed"])
    micro = micro or W["micro_batch"]; accum = accum or W["grad_accum"]; out = out or W["out"]
    dev = cfg["student"]["device"]
    st = Student(cfg, dev)
    if W.get("grad_ckpt"):
        st.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    st.llm.config.use_cache = False
    st.llm.train()
    prep, tok = VisionPrep(cfg), st.tok
    ds = JsonlImages(cfg["data"]["warmup_jsonl"], cfg["data"]["coco_root"], limit)
    ptxt = st.prompt_text(W["instruction"])
    p_ids = tok(ptxt, add_special_tokens=False)["input_ids"]
    end = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    def collate(rows):
        ids, labels = [], []
        for r in rows:
            c = tok(r["caption"].strip(), add_special_tokens=False)["input_ids"] + end
            x = (p_ids + c)[: W["max_len"]]
            ids.append(x); labels.append(([-100] * len(p_ids) + c)[: W["max_len"]])
        L = max(map(len, ids)); pad = tok.pad_token_id
        I = torch.full((len(ids), L), pad); Y = torch.full((len(ids), L), -100); M = torch.zeros((len(ids), L), dtype=torch.long)
        for i, (a, b) in enumerate(zip(ids, labels)):
            I[i, :len(a)] = torch.tensor(a); Y[i, :len(b)] = torch.tensor(b); M[i, :len(a)] = 1
        return I, Y, M, prep([r["pil"] for r in rows])

    dl = DataLoader(ds, batch_size=micro, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True,
                    generator=torch.Generator().manual_seed(cfg["seed"]))
    total = math.ceil(len(dl) * W["epochs"] / accum)
    opt = torch.optim.AdamW(st.projector.parameters(), lr=W["lr"], weight_decay=W["weight_decay"])
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(W["warmup_ratio"] * total)), total)
    os.makedirs(out, exist_ok=True); lf = open(f"{out}/train_log.jsonl", "a")
    log({"stage": "warmup", "n_data": len(ds), "micro": micro, "accum": accum, "total_steps": total, "prompt_tokens": len(p_ids)}, lf)
    step, t0, acc_loss, k = 0, time.time(), 0.0, 0
    for ep in range(W["epochs"]):
        for I, Y, M, vis in dl:
            I, Y, M = I.to(dev), Y.to(dev), M.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = st.embed(I, vis)
                logits = st.llm(inputs_embeds=emb, attention_mask=M).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), Y[:, 1:].reshape(-1), ignore_index=-100)
            (loss / accum).backward(); acc_loss += loss.item(); k += 1
            if k % accum: continue
            gn = torch.nn.utils.clip_grad_norm_(st.projector.parameters(), W["grad_clip"]).item()
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
            dt = time.time() - t0; t0 = time.time()
            log({"step": step, "loss": round(acc_loss / accum, 4), "grad_norm": round(gn, 3), "lr": sch.get_last_lr()[0], "sec_per_step": round(dt, 2), "peak_mem_gb": peak_mem(dev)}, lf)
            acc_loss = 0.0
            if step % W["save_every"] == 0:
                save_projector(st, f"{out}/step{step}", {"step": step})
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    save_projector(st, f"{out}/final", {"step": step, "config": config_path})
    log({"done": True, "steps": step, "out": f"{out}/final"}, lf)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True); a.add_argument("--max_steps", type=int); a.add_argument("--micro", type=int)
    a.add_argument("--accum", type=int); a.add_argument("--limit", type=int); a.add_argument("--out")
    x = a.parse_args()
    run_warmup(x.config, x.max_steps, x.micro, x.accum, x.limit, x.out)
