from data_pre import data

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from llama_add import add_LlamaForCausalLM
import numpy as np

from model import load_model, load_tokenizer
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
    def __init__(self, model, pos, model_path, save_path, dataset, device="cpu"):
        self.data = data(dataset)
        self.data_name = dataset
        self.pos = pos
        self.device = device
        self.model_name = model
        self.model = load_model(model_path, device=self.device)
        self.model.eval()
        self.tokenizer = load_tokenizer(model_path)

        self.path = "".join([save_path, dataset, "-", model, "-", pos,".json"])

    def infer(self, SV):
        SV = SV.numpy()
        SV_norm = np.linalg.norm(SV)

        
        res = []

        for example in tqdm(self.data, desc="infering"):

            tmp_dic = {}

            with torch.no_grad():
                query1 = self.tokenizer(prompt_dict["harm"].format_map(example) + example["chosen"], return_tensors='pt').to(
                    self.device)
                query2 = self.tokenizer(prompt_dict["harm"].format_map(example) + example["rejected"], return_tensors='pt').to(
                    self.device)
                if self.pos == "last":
                    pred1 = self.model(**query1)['hidden_states'][-1][0, -1, :].cpu().numpy()
                    pred2 = self.model(**query2)['hidden_states'][-1][0, -1, :].cpu().numpy()
                elif self.pos == "mean":
                    pred1 = torch.mean(self.model(**query1)['hidden_states'][-1][0, :, :], 0).cpu().numpy()
                    pred2 = torch.mean(self.model(**query2)['hidden_states'][-1][0, :, :], 0).cpu().numpy()

                margin = (pred1 - pred2).dot(SV) / (SV_norm)
            tmp_dic["input"] = example["input"]
            tmp_dic["chosen"] = example["chosen"]
            tmp_dic["rejected"] = example["rejected"]

            tmp_dic["margin"] = margin.tolist()
            tmp_dic["generator"] = self.model_name

            res.append(tmp_dic)

        self.res_save(res)
        print("infer finished and saved")


    def res_save(self, res):
        with open(self.path, "w", encoding="utf-8") as f0:
            js = json.dumps(res, ensure_ascii=False)
            f0.write(js)
            f0.close()

def read_json(save_path):
    with open(save_path, "r", encoding="utf-8") as f1:
        data_res = []
        for text in tqdm(f1):
            text = json.loads(text)
            for line in tqdm(text):
                data_res.append(line["margin"])

        f1.close()
    return data_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--model_path_use', default=True, help="use model path")
    parser.add_argument('--model_path', default="/xxx/", help="model path")
    parser.add_argument('--model', default='Llama-2-7b', help='Name(path) of LLM')
    parser.add_argument('--pos', default='last', help='which token as embed')  
    parser.add_argument('--save_path', default='/xxx/', help='Name(path) of LLM')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--dataset', default='pku_train', help='datasets')
    parser.add_argument("--device_ids", type=list, default=[0])
    args = parser.parse_args()

    if args.model_path_use == True:
        model_path = "".join([args.model_path, args.model])
    else:
        model_path = args.model

    SV = torch.load("".join([args.save_path,"SV","-",args.model,".pt"]))

    Gener = infer_res(model=args.model, pos=args.pos, model_path=model_path, save_path=args.save_path, dataset=args.dataset, device=args.device)
    Gener.infer(SV)
   

