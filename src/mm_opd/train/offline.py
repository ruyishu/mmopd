"""Baselines on teacher responses (same questions as OPD): SFT (CE) or offline KD (teacher-forced
per-token reverse KL(student||teacher), forward KL logged). Projector + LoRA, init from warmup projector,
same lr / schedule / steps / global batch as OPD (cfg['opd']).
python -m mm_opd.train.offline --config CFG --mode sft|kd [--max_steps N --micro M --accum A --limit K --out DIR]"""
import argparse, os, time
import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup
from mm_opd.imgdata import JsonlImages, VisionPrep
from mm_opd.model.student import Student
from mm_opd.train.common import load_cfg, log, peak_mem, save_projector, seed_all
from mm_opd.train.opd import Teacher, cat_vis, chunked_reverse_kl


@torch.no_grad()
def forward_kl(S, T, chunk=256):
    tot = 0.0
    for i in range(0, S.shape[0], chunk):
        s = torch.log_softmax(S[i:i + chunk].float(), -1); t = torch.log_softmax(T[i:i + chunk].float(), -1)
        tot += (t.exp() * (t - s)).sum().item()
    return tot


def run_offline(config_path, mode, max_steps=None, micro=None, accum=None, limit=None, out=None):
    cfg = load_cfg(config_path); O = cfg["opd"]; seed_all(cfg["seed"])
    micro = micro or O["micro_batch"]; accum = accum or O["grad_accum"]
    out = out or f"ckpt/demo_{mode}"
    sdev, tdev = cfg["student"]["device"], cfg["teacher"]["device"]
    st = Student(cfg, sdev)
    st.projector.load_state_dict(torch.load(O["init_projector"])); print("loaded projector", O["init_projector"])
    lc = cfg["student"]["lora"]
    st.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    st.llm.config.use_cache = False
    st.llm = get_peft_model(st.llm, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    st.llm.train()
    teacher = Teacher(cfg["teacher"]["path"], tdev) if mode == "kd" else None
    tok, prep = st.tok, VisionPrep(cfg); tok.padding_side = "left"
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ds = JsonlImages(cfg["data"]["teacher_resp_jsonl"], cfg["data"]["coco_root"], limit)
    lora_params = [p for p in st.llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": O["lr_lora"]}, {"params": list(st.projector.parameters()), "lr": O["lr_projector"]}], weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, O["warmup_steps"], O["steps"])
    os.makedirs(out, exist_ok=True); lf = open(f"{out}/train_log.jsonl", "a")
    g = torch.Generator().manual_seed(cfg["seed"]); order, ptr = torch.randperm(len(ds), generator=g).tolist(), 0
    log({"stage": mode, "n_data": len(ds), "micro": micro, "accum": accum, "steps": O["steps"]}, lf)
    steps = min(O["steps"], max_steps or O["steps"]); T0 = time.time()
    for step in range(1, steps + 1):
        t0 = time.time(); tot_loss, tot_fkl, tot_tok, t_tea = 0.0, 0.0, 0, 0.0
        for _ in range(accum):
            rows = []
            for _ in range(micro):
                if ptr >= len(order): order, ptr = torch.randperm(len(ds), generator=g).tolist(), 0
                rows.append(ds[order[ptr]]); ptr += 1
            enc = tok([st.prompt_text(r["question"]) for r in rows], add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=O["prompt_max_len"])
            P_ids, P_m = enc["input_ids"].to(sdev), enc["attention_mask"].to(sdev)
            ans = [tok(r["response"], add_special_tokens=False)["input_ids"][:O["max_new_tokens"] - 1] + [im_end] for r in rows]
            G = max(map(len, ans))
            gen = torch.full((len(ans), G), tok.pad_token_id, device=sdev); gm = torch.zeros_like(gen)
            for i, a in enumerate(ans): gen[i, :len(a)] = torch.tensor(a, device=sdev); gm[i, :len(a)] = 1
            ids = torch.cat([P_ids, gen], 1); am = torch.cat([P_m, gm], 1); pos = (am.cumsum(1) - 1).clamp(min=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = st.embed(ids, prep([r["pil"] for r in rows]))
                s_logits = st.llm(inputs_embeds=emb, attention_mask=am, position_ids=pos, logits_to_keep=G + 1).logits[:, :-1]
            S = torch.cat([s_logits[i, :len(a)] for i, a in enumerate(ans)]); n = S.shape[0]
            if mode == "sft":
                tgt = torch.tensor([t for a in ans for t in a], device=sdev)
                loss_sum = torch.nn.functional.cross_entropy(S.float(), tgt, reduction="sum")
            else:
                tt = time.time()
                T = torch.cat([teacher.answer_logits(r["pil"], r["question"], a).to(sdev) for r, a in zip(rows, ans)]); t_tea += time.time() - tt
                loss_sum = chunked_reverse_kl(S, T, O["kl_temperature"], O["kl_chunk"]).sum()
                tot_fkl += forward_kl(S.detach(), T); del T
            (loss_sum / n / accum).backward()
            tot_loss += loss_sum.item(); tot_tok += n
            del s_logits, S, emb, loss_sum
        gn = torch.nn.utils.clip_grad_norm_(lora_params + list(st.projector.parameters()), O["grad_clip"]).item()
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        d = {"step": step, ("ce" if mode == "sft" else "reverse_kl"): round(tot_loss / tot_tok, 4), "grad_norm": round(gn, 3), "lr_lora": sch.get_last_lr()[0],
             "tokens": tot_tok, "sec_per_step": round(time.time() - t0, 1), "elapsed_min": round((time.time() - T0) / 60, 1), "peak_mem_student_gb": peak_mem(sdev)}
        if mode == "kd": d.update({"forward_kl": round(tot_fkl / tot_tok, 4), "teacher_sec": round(t_tea, 1), "peak_mem_teacher_gb": peak_mem(tdev)})
        log(d, lf)
        if step % O["save_every"] == 0: save(st, f"{out}/step{step}", step)
    save(st, f"{out}/final", steps)
    log({"done": True, "steps": steps, "train_minutes": round((time.time() - T0) / 60, 1)}, lf)


def save(st, d, step):
    save_projector(st, d, {"step": step}); st.llm.save_pretrained(f"{d}/lora")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True); a.add_argument("--mode", choices=["sft", "kd"], required=True)
    a.add_argument("--max_steps", type=int); a.add_argument("--micro", type=int); a.add_argument("--accum", type=int)
    a.add_argument("--limit", type=int); a.add_argument("--out")
    x = a.parse_args(); run_offline(x.config, x.mode, x.max_steps, x.micro, x.accum, x.limit, x.out)
