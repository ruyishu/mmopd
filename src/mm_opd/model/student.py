"""Student: text LLM (Qwen3) + vision front-end + trainable projector.

vision.kind == "qwen3vl_vit": frozen Qwen3-VL visual tower (patch embed + blocks + merger,
deepstack ignored) -> projector (mlp/linear, vit_out_dim -> hidden).
vision.kind == "patch": raw pixel patches -> PatchProjector (encoder-free).
Image features replace the <|image_pad|> token embeddings (embedding replacement).
"""
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from mm_opd.model.patch_projector import PatchProjector

VISION_START, IMAGE_PAD, VISION_END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"


def modules_for_stage(stage: str) -> tuple[str, ...]:
    if stage == "warmup":
        return ("projector",)
    if stage == "opd":
        return ("projector", "llm.lora")
    raise ValueError(f"unknown stage: {stage!r}")


def make_projector(kind: str, din: int, dout: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(din, dout)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(din, dout), nn.GELU(), nn.Linear(dout, dout))
    raise ValueError(kind)


def load_qwen3vl_visual(path: str, dtype=torch.bfloat16) -> nn.Module:
    from transformers import Qwen3VLForConditionalGeneration
    full = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map="cpu")
    vis = full.model.visual
    full.model.visual = None
    del full
    return vis


class Student(nn.Module):
    def __init__(self, cfg: dict, device: str):
        super().__init__()
        self.cfg, self.device_ = cfg, device
        v = cfg["vision"]
        self.kind = v["kind"]
        self.tok = AutoTokenizer.from_pretrained(cfg["student"]["path"])
        self.llm = AutoModelForCausalLM.from_pretrained(cfg["student"]["path"], dtype=torch.bfloat16).to(device)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        H = cfg["student"]["hidden_size"]
        if self.kind == "qwen3vl_vit":
            self.visual = load_qwen3vl_visual(v["vit_from"]).to(device).eval()
            for p in self.visual.parameters():
                p.requires_grad_(False)
            self.projector = make_projector(v["projector"], v["vit_out_dim"], H)
            self.n_img = v["num_image_tokens"]
        elif self.kind == "patch":
            pc = v["patch"]
            self.visual = None
            self.projector = PatchProjector(pc["patch_size"], H, kind=pc["projector"])
            self.n_img = pc["num_image_tokens"]
        else:
            raise ValueError(self.kind)
        self.projector.to(device=device, dtype=torch.float32)
        self.image_pad_id = self.tok.convert_tokens_to_ids(IMAGE_PAD)

    # ---- prompts ----
    def image_block(self) -> str:
        return VISION_START + IMAGE_PAD * self.n_img + VISION_END

    def prompt_text(self, question: str) -> str:
        msgs = [{"role": "user", "content": self.image_block() + question}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # ---- vision ----
    def encode_images(self, vis_inputs: dict) -> Tensor:
        """Returns (B*n_img, H) float32 features in raster order."""
        if self.kind == "qwen3vl_vit":
            with torch.no_grad():
                out = self.visual(vis_inputs["pixel_values"].to(self.device_, torch.bfloat16),
                                  grid_thw=vis_inputs["image_grid_thw"].to(self.device_))
                feats = out.pooler_output if hasattr(out, "pooler_output") else out[0]
            feats = feats.float()
        else:
            feats = vis_inputs["pixels"].to(self.device_, torch.float32)
        z = self.projector(feats)
        return z.reshape(-1, z.shape[-1])

    def embed(self, input_ids: Tensor, vis_inputs: dict) -> Tensor:
        emb = self.llm.get_input_embeddings()(input_ids)
        img = self.encode_images(vis_inputs).to(emb.dtype)
        mask = input_ids == self.image_pad_id
        if int(mask.sum()) != img.shape[0]:
            raise ValueError(f"image_pad count {int(mask.sum())} != image features {img.shape[0]}")
        return emb.masked_scatter(mask.unsqueeze(-1), img)

    def trainable_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.projector.state_dict().items()}
