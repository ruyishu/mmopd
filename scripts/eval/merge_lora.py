"""Merge LoRA adapter into Qwen3-1.7B for vLLM text eval. Usage: python merge_lora.py BASE LORA_DIR OUT"""
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
b, l, o = sys.argv[1:4]
m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(b, dtype=torch.bfloat16), l).merge_and_unload()
m.save_pretrained(o); AutoTokenizer.from_pretrained(b).save_pretrained(o); print("merged ->", o)
