import os
import unittest

from os import path
import torch
from pipeformers import LlamaForCausalLM
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer


tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(torch.device("cuda"))

class TestLlama(unittest.TestCase):
    def test_logits(self):
        config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        device = torch.device("cuda")

        state_dict = torch.load(".state_dicts/meta-llama--Meta-Llama-3.1-8B-Instruct.pt", weights_only=True)
        model = LlamaForCausalLM(config=config).to(device)
        model.load_state_dict(state_dict)

        inputs = tokenizer("Hello, world!", return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        test_logits = model((input_ids, attention_mask)).to("cpu")
        os.makedirs(".snapshots/meta-llama--Meta-Llama-3.1-8B-Instruct", exist_ok=True)
        torch.save(test_logits, ".snapshots/meta-llama--Meta-Llama-3.1-8B-Instruct/test_logits.pt")

        del model
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", state_dict=state_dict, torch_dtype=torch.bfloat16).to(device)
        reference_logits = model(input_ids, attention_mask).logits.to("cpu").to(test_logits.dtype)
        torch.save(reference_logits, ".snapshots/meta-llama--Meta-Llama-3.1-8B-Instruct/reference_logits.pt")

        self.assertTrue(torch.equal(test_logits, reference_logits))


