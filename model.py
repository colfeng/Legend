import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModel, AutoModelForCausalLM

import os


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    return tokenizer

def load_model(model_name, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"


    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states = True,trust_remote_code=True)
    model.eval()
    model.to(device)
    print('The device of the LLM:', model.device)
    return model









