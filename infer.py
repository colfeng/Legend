# https://github.com/davidbau/baukit
from data_pre import data

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import numpy as np

from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt

prompt_dict = {
    "harm": (
        "Below is an instruction that describes a task. "
        "Write a response that completes the request.\n\n"
        "### Instruction:\n{input}\n\n### Response:"
    )}

class  infer_res():
    def __init__(self, model, model_path, save_path, dataset, device="cpu"):
        self.data = data(dataset)
        self.data_name = dataset
        self.device = device

        self.model_name = model
        self.model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.eos_token_id != None:
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            print("may need eos token")

        self.path = "".join([save_path, dataset, "-", model, "-", "hm",  ".json"])


    def infer(self):
        generation_kwargs = {"do_sample": False, "max_new_tokens": 1024}
        generation_kwargs['eos_token_id'] = self.eos_token_id
        generation_kwargs['pad_token_id'] = self.eos_token_id

        with torch.no_grad():
            res = []
            for example in tqdm(self.data, desc="infering"):

                tmp_dic = {}

                query = self.tokenizer(prompt_dict["harm"].format_map(example), return_tensors='pt').to(self.device)

                pred = self.model.generate(**query, **generation_kwargs)
                query_len = query["input_ids"].shape[1]
                response = self.tokenizer.decode(pred.cpu()[0][query_len:], skip_special_tokens=True)
                tmp_dic["input"] = example["input"]
                tmp_dic["output"] = response
                tmp_dic["generator"] = self.model_name

                res.append(tmp_dic)

        self.res_save(res)
        print("infer finished and saved")


    def res_save(self, res):
        with open(self.path, "w", encoding="utf-8") as f0:
            js = json.dumps(res, ensure_ascii=False)
            f0.write(js)
            f0.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--model_path_use', default=True, help="use model path")
    parser.add_argument('--model_path', default="/xxx/", help="model path")
    parser.add_argument('--model', default='Llama-2-7b', help='Name(path) of LLM')
    parser.add_argument('--save_path', default='/xxx/', help='Name(path) of LLM')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--dataset', default='hb', help='datasets')
    parser.add_argument("--device_ids", type=list, default=[0])
    args = parser.parse_args()

    if args.model_path_use == True:
        model_path = "".join([args.model_path, args.model])
    else:
        model_path = args.model


    Gener = infer_res(model=args.model, model_path=model_path, save_path=args.save_path, dataset=args.dataset, device=args.device)
    Gener.infer()




