from datasets import load_dataset
from torch.utils.data import random_split
import torch

def data(dataset_name):
    if dataset_name == "hb":
        dataset = load_dataset('csv', data_files={'train': "harmful_behaviors.csv"})

        dataset_val = dataset["train"]


        def split_prompt_and_responses(sample):

            return {
                "input": sample["goal"],
                "output": sample["target"],
            }

        dataset_val = dataset_val.map(split_prompt_and_responses, remove_columns=["goal","target"])

    elif dataset_name == "hb_res":
        dataset = load_dataset('json', data_files={'train': "hb_res.json"})

        dataset_val = dataset["train"]


        def split_prompt_and_responses(sample):

            return {
                "input": sample["input"],
                "output": sample["output"],
            }

        dataset_val = dataset_val.map(split_prompt_and_responses, remove_columns=["generator"])

    elif dataset_name == "harm_train":
        dataset = load_dataset('json', data_files={'train': "harmless_train.jsonl"})

        dataset_val = dataset["train"]


        def split_prompt_and_responses(sample):
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "input": prompt,
                "chosen": sample["chosen"][len(prompt):],
                "rejected": sample["rejected"][len(prompt):],
            }

        dataset_val = dataset_val.map(split_prompt_and_responses)

    elif dataset_name == "harm_test":
        dataset = load_dataset('json', data_files={'train': "harmless_test.jsonl"})

        dataset_val = dataset["train"]


        def split_prompt_and_responses(sample):
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "input": prompt,
                "chosen": sample["chosen"][len(prompt):],
                "rejected": sample["rejected"][len(prompt):],
            }

        dataset_val = dataset_val.map(split_prompt_and_responses)



    elif dataset_name == "pku_train":
        dataset = load_dataset('json', data_files={
            'train': "pku_train.json"})

        dataset_val = dataset["train"]

        def split_prompt_and_responses(sample):

            return {
                "input": sample["input"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
            }

        dataset_val = dataset_val.map(split_prompt_and_responses)

    elif dataset_name == "pku_test":
        dataset = load_dataset('json', data_files={
            'train': "pku_test.json"})

        dataset_val = dataset["train"]

        def split_prompt_and_responses(sample):

            return {
                "input": sample["input"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
            }

        dataset_val = dataset_val.map(split_prompt_and_responses)



if __name__ == "__main__":
    data("harm")