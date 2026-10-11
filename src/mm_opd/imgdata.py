"""Datasets + image preprocessing for the student (fixed resolution) and teacher (original image)."""
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_jsonl(p, limit=None):
    out = []
    with open(p) as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


class JsonlImages(Dataset):
    def __init__(self, path, coco_root, limit=None):
        self.rows, self.root = read_jsonl(path, limit), coco_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = dict(self.rows[i])
        r["pil"] = Image.open(f"{self.root}/{r['image']}").convert("RGB")
        return r


class VisionPrep:
    """Student-side image preprocessing at a fixed resolution -> constant token count."""

    def __init__(self, cfg):
        v = cfg["vision"]
        self.kind = v["kind"]
        if self.kind == "qwen3vl_vit":
            from transformers import AutoProcessor
            self.size = v["image_size"]
            self.ip = AutoProcessor.from_pretrained(v["vit_from"]).image_processor
        else:
            self.size = v["patch"]["image_size"]

    def __call__(self, pils):
        ims = [p.resize((self.size, self.size), Image.BICUBIC) for p in pils]
        if self.kind == "qwen3vl_vit":
            o = self.ip(images=ims, return_tensors="pt")
            return {"pixel_values": o["pixel_values"], "image_grid_thw": o["image_grid_thw"]}
        import numpy as np
        x = torch.stack([torch.from_numpy(np.asarray(im, dtype="float32") / 255.0).permute(2, 0, 1) for im in ims])
        return {"pixels": x}
