import json, os, random, time
import numpy as np
import torch
import yaml


def load_cfg(path):
    return yaml.safe_load(open(path))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def log(d, f=None):
    s = json.dumps(d, ensure_ascii=False)
    print(s, flush=True)
    if f:
        f.write(s + "\n"); f.flush()


def peak_mem(dev):
    return round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)


def save_projector(student, out_dir, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(student.trainable_state(), f"{out_dir}/projector.pt")
    if extra:
        json.dump(extra, open(f"{out_dir}/meta.json", "w"), indent=2)
