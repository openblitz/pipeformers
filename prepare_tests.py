import os
from transformers import AutoModelForCausalLM
import torch

if __name__ == "__main__":
    os.makedirs(".state_dicts", exist_ok=True)
    os.makedirs(".snapshots", exist_ok=True)

    llama = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    torch.save(llama.state_dict(), ".state_dicts/meta-llama--Meta-Llama-3.1-8B-Instruct.pt")