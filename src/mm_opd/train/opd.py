"""Stage 2: on-policy distillation.
Student (projector + LoRA) samples answers on (image, question); frozen Qwen3-VL teacher scores the
student's real answer tokens with its own chat template + processor on the original image.
Loss = per-token reverse KL(student || teacher) over full vocab, chunked.
python -m mm_opd.train.opd --config configs/demo_1p7b.yaml [--max_steps N --micro M --accum A --limit K --out DIR --no_init]
"""
import argparse, os, time
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

from mm_opd.imgdata import JsonlImages, VisionPrep
from mm_opd.model.student import Student
from mm_opd.train.common import load_cfg, log, peak_mem, save_projector, seed_all


def _kl_chunk(s_logits, t_logits, temp):
    s = torch.log_softmax(s_logits.float() / temp, -1)
    t = torch.log_softmax(t_logits.float() / temp, -1)
    return (s.exp() * (s - t)).sum(-1)


def chunked_reverse_kl(s_logits, t_logits, temp=1.0, chunk=256):
    """s_logits, t_logits: (N, V) for N answer tokens. Returns per-token KL (N,)."""
    out = []
    for i in range(0, s_logits.shape[0], chunk):
        out.append(checkpoint(_kl_chunk, s_logits[i:i + chunk], t_logits[i:i + chunk], temp, use_reentrant=False))
    return torch.cat(out)


class Teacher:
    def __init__(self, path, device):
        self.dev = device
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def answer_logits(self, pil, question, ans_ids):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = self.proc(text=[text], images=[pil], return_tensors="pt")
        ids = torch.cat([x["input_ids"], torch.tensor([ans_ids])], 1).to(self.dev)
        am = torch.ones_like(ids)
        L = len(ans_ids)
        extra = {}
        if "mm_token_type_ids" in x:  # needed for M-RoPE in recent transformers; answer tokens are text (0)
            extra["mm_token_type_ids"] = torch.cat([x["mm_token_type_ids"], torch.zeros((1, L), dtype=x["mm_token_type_ids"].dtype)], 1).to(self.dev)
        out = self.model(input_ids=ids, attention_mask=am, **extra, pixel_values=x["pixel_values"].to(self.dev, torch.bfloat16),
                         image_grid_thw=x["image_grid_thw"].to(self.dev), logits_to_keep=L + 1)
        return out.logits[0, :-1]  # (L, V) predicting ans tokens


def run_opd(config_path, max_steps=None, micro=None, accum=None, limit=None, out=None, no_init=False):
    cfg = load_cfg(config_path); O = cfg["opd"]
    seed_all(cfg["seed"])
    micro = micro or O["micro_batch"]; accum = accum or O["grad_accum"]; out = out or O["out"]
    sdev, tdev = cfg["student"]["device"], cfg["teacher"]["device"]
    backend = O.get("rollout_backend", "hf")
    vllm_eng = None
    if backend == "vllm":  # init first so its memory check sees a clean GPU
        from vllm import LLM
        lc0 = cfg["student"]["lora"]
        vllm_eng = LLM(model=cfg["student"]["path"], dtype="bfloat16", enable_lora=True, max_lora_rank=lc0["r"], max_loras=1,
                       enable_prompt_embeds=True, gpu_memory_utilization=O.get("vllm_gpu_util", 0.12),
                       max_model_len=O["total_max_len"], enforce_eager=O.get("vllm_enforce_eager", False), seed=cfg["seed"])
    st = Student(cfg, sdev)
    if not no_init and O.get("init_projector") and os.path.exists(O["init_projector"]):
        st.projector.load_state_dict(torch.load(O["init_projector"]))
        print("loaded projector", O["init_projector"])
    elif not no_init:
        raise FileNotFoundError(f"init_projector missing: {O.get('init_projector')} (pass --no_init for smoke tests)")
    lc = cfg["student"]["lora"]
    if O.get("grad_ckpt"):
        st.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    st.llm = get_peft_model(st.llm, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                               target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    teacher = Teacher(cfg["teacher"]["path"], tdev)
    tok, prep = st.tok, VisionPrep(cfg)
    tok.padding_side = "left"
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id, tok.pad_token_id}
    ds = JsonlImages(cfg["data"]["opd_jsonl"], cfg["data"]["coco_root"], limit)
    lora_params = [p for n, p in st.llm.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": O["lr_lora"]},
                             {"params": list(st.projector.parameters()), "lr": O["lr_projector"]}], weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, O["warmup_steps"], O["steps"])
    os.makedirs(out, exist_ok=True); lf = open(f"{out}/train_log.jsonl", "a")
    g = torch.Generator().manual_seed(cfg["seed"])
    n_lora = sum(p.numel() for p in lora_params)
    log({"stage": "opd", "n_data": len(ds), "micro": micro, "accum": accum, "steps": O["steps"], "lora_params": n_lora}, lf)
    order, ptr = torch.randperm(len(ds), generator=g).tolist(), 0
    steps = min(O["steps"], max_steps or O["steps"])
    for step in range(1, steps + 1):
        t0 = time.time(); tot_kl, tot_tok, lens, empties, t_gen, t_tea = 0.0, 0, [], 0, 0.0, 0.0
        # ---- one batched rollout for the whole global step (micro*accum prompts, left padded) ----
        rows = []
        for _ in range(micro * accum):
            if ptr >= len(order):
                order, ptr = torch.randperm(len(ds), generator=g).tolist(), 0
            rows.append(ds[order[ptr]]); ptr += 1
        texts = [st.prompt_text(r["question"]) for r in rows]
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=O["prompt_max_len"])
        P_all, M_all = enc["input_ids"].to(sdev), enc["attention_mask"].to(sdev)
        vis_all = [prep([r["pil"]]) for r in rows]
        tg = time.time(); t_sync = 0.0
        max_new = min(O["max_new_tokens"], O["total_max_len"] - P_all.shape[1])
        if backend == "vllm":
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            ts = time.time()
            lora_dir = f"{out}/_vllm_lora/step{step}"
            st.llm.save_pretrained(lora_dir)
            if step > 1:
                import shutil; shutil.rmtree(f"{out}/_vllm_lora/step{step-1}", ignore_errors=True)
            t_sync = time.time() - ts
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                emb = st.embed(P_all, cat_vis(vis_all))  # embed_tokens has no LoRA -> identical for vLLM
            prompts = [{"prompt_embeds": emb[i][M_all[i].bool()].to(torch.bfloat16).cpu()} for i in range(emb.shape[0])]
            del emb
            sp = SamplingParams(temperature=O["rollout_temperature"], top_p=O["rollout_top_p"], max_tokens=max_new)
            outs = vllm_eng.generate(prompts, sp, lora_request=LoRARequest(f"s{step}", step, lora_dir), use_tqdm=False)
            gen_all = [list(o.outputs[0].token_ids) for o in outs]
        else:
            st.llm.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                emb = st.embed(P_all, cat_vis(vis_all))
                g_t = st.llm.generate(inputs_embeds=emb, attention_mask=M_all, do_sample=True, temperature=O["rollout_temperature"],
                                      top_p=O["rollout_top_p"], top_k=0, max_new_tokens=max_new, use_cache=True,
                                      pad_token_id=tok.pad_token_id)
            del emb
            gen_all = g_t.tolist()
        t_gen = time.time() - tg - t_sync
        st.llm.train()
        ans_all = []
        for row in gen_all:
            n = len(row)
            for j, t in enumerate(row):
                if t in eos_ids:
                    n = j + 1; break
            ans_all.append(row[:n]); lens.append(n)
            if n <= 1: empties += 1
        if O.get("log_rollouts_every", 0) and (step == 1 or step % O["log_rollouts_every"] == 0):
            import json as _j
            with open(f"{out}/rollouts.jsonl", "a") as _f:
                for _r, _a in list(zip(rows, ans_all))[:O.get("log_rollouts_n", 4)]:
                    _f.write(_j.dumps({"step": step, "image": _r["image"], "question": _r["question"], "n_tokens": len(_a),
                                       "text": tok.decode(_a, skip_special_tokens=False)}, ensure_ascii=False) + "\n")
        # ---- forward/backward in micro-batches ----
        for k in range(accum):
            sl = slice(k * micro, (k + 1) * micro)
            mrows, ans = rows[sl], ans_all[sl]
            G = max(len(a) for a in ans)
            P_ids, P_m = P_all[sl], M_all[sl]
            gen = torch.full((len(ans), G), tok.pad_token_id, device=sdev); gm = torch.zeros_like(gen)
            for i, a in enumerate(ans):
                gen[i, :len(a)] = torch.tensor(a, device=sdev); gm[i, :len(a)] = 1
            tt = time.time()
            t_logits = [teacher.answer_logits(r["pil"], r["question"], a).to(sdev) for r, a in zip(mrows, ans)]
            t_tea += time.time() - tt
            ids = torch.cat([P_ids, gen], 1); am = torch.cat([P_m, gm], 1)
            pos = (am.cumsum(1) - 1).clamp(min=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = st.embed(ids, cat_vis(vis_all[sl]))
                s_logits = st.llm(inputs_embeds=emb, attention_mask=am, position_ids=pos, logits_to_keep=G + 1).logits[:, :-1]
            S = torch.cat([s_logits[i, :len(a)] for i, a in enumerate(ans)])
            T = torch.cat(t_logits)
            kl = chunked_reverse_kl(S, T, O["kl_temperature"], O["kl_chunk"])
            ntok = kl.numel()
            (kl.sum() / ntok / accum).backward()
            tot_kl += kl.sum().item(); tot_tok += ntok
            del s_logits, S, T, t_logits, kl, emb
        params = lora_params + list(st.projector.parameters())
        gn = torch.nn.utils.clip_grad_norm_(params, O["grad_clip"]).item()
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        log({"step": step, "reverse_kl": round(tot_kl / max(tot_tok, 1), 4), "grad_norm": round(gn, 3), "lr_lora": sch.get_last_lr()[0],
             "mean_ans_len": round(sum(lens) / len(lens), 1), "max_ans_len": max(lens), "empty_frac": empties / len(lens),
             "sec_per_step": round(time.time() - t0, 1), "gen_sec": round(t_gen, 1), "sync_sec": round(t_sync, 2), "backend": backend, "teacher_sec": round(t_tea, 1),
             "peak_mem_student_gb": peak_mem(sdev), "peak_mem_teacher_gb": peak_mem(tdev)}, lf)
        if step % O["save_every"] == 0:
            save(st, f"{out}/step{step}", step)
    save(st, f"{out}/final", steps)
    log({"done": True, "steps": steps}, lf)


def cat_vis(vs):
    return {k: torch.cat([v[k] for v in vs], 0) for k in vs[0]}


def save(st, d, step):
    save_projector(st, d, {"step": step})
    st.llm.save_pretrained(f"{d}/lora")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True); a.add_argument("--max_steps", type=int); a.add_argument("--micro", type=int)
    a.add_argument("--accum", type=int); a.add_argument("--limit", type=int); a.add_argument("--out"); a.add_argument("--no_init", action="store_true")
    x = a.parse_args()
    run_opd(x.config, x.max_steps, x.micro, x.accum, x.limit, x.out, x.no_init)
