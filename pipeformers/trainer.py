from .models import MODELS

import deepspeed
from deepspeed.monitor.monitor import WandbMonitor
from datasets import Dataset, load_dataset
from deepspeed.pipe import PipelineModule
import json
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
import torch

import argparse
import os
import wandb
from typing import Literal, Optional


class PipeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        sequence_length: int,
        split: Literal["train", "valid"] = "train",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.split = split
        self.features = dataset.features
        self.mode = "pretrain"
        
        if "content" in self.features:
            self.pretrain_column = "content"
        else:
            self.pretrain_column = "text"

        self.dataset = self.dataset.filter(
            lambda x: x[self.pretrain_column] != "",
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]

        row = {"text": "Hello, world!"}

        if self.mode == "pretrain":
            text = row[self.pretrain_column]
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = inputs["input_ids"].squeeze(dim=0)
            attention_mask = inputs["attention_mask"].squeeze(dim=0)
            labels = input_ids.clone()

            if input_ids.numel() <= 2:
                raise ValueError(f"Empty input: `{text}` @ {idx}")

            if len(input_ids) < self.sequence_length:
                # right padding is required as flash-attn's pad_input function pads on the right

                input_ids = torch.cat(
                    [
                        input_ids,
                        self.tokenizer.pad_token_id * torch.ones(self.sequence_length - len(input_ids), dtype=torch.long),
                    ]
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros(self.sequence_length - len(attention_mask), dtype=torch.long),
                    ]
                )
                labels = torch.cat(
                    [
                        labels,
                        -100 * torch.ones(self.sequence_length - len(labels), dtype=torch.long),
                    ]
                )
            elif len(input_ids) > self.sequence_length:
                input_ids = input_ids[:self.sequence_length]
                attention_mask = attention_mask[:self.sequence_length]
                labels = labels[:self.sequence_length]
        else:
            raise ValueError(f"Invalid dataset mode: {self.mode}")

        return (
            tuple([
                input_ids,
                attention_mask,
            ]),
            tuple([labels]),
        )

def loss_fn(logits: torch.Tensor, data: tuple[torch.Tensor]):
    labels, = data

    logits = logits[:, :-1, :]
    labels = labels[:, 1:]  # next token prediction

    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

    return loss


def save_model(
    engine: deepspeed.runtime.engine.DeepSpeedEngine,
    output_dir: str,
    prefix: str,
):
    if engine.zero_optimization_partition_weights():
        state_dict = engine._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = engine.state_dict()

    model_dir = os.path.join(output_dir, prefix)
    os.makedirs(model_dir, exist_ok=True)

    torch.save(state_dict, os.path.join(model_dir, f"rank{torch.distributed.get_rank()}.pt"))


def main(
    base_model: str,
    dataset: str,
    dataset_config: Optional[str],
    dataset_split: str,
    deepspeed_config: str,
    epochs: int,
    output_dir: str,
    pipeline_stages: int,
    sequence_length: int,
    state_dict: Optional[str] = None,
    validation_stride: int = 1000,
    wandb_logging: bool = False,
):
    with open(deepspeed_config, "r") as f:
        config_params = json.load(f)

    dataset_dict = load_dataset(dataset, name=dataset_config, split="train").train_test_split(test_size=dataset_split)

    config = AutoConfig.from_pretrained(base_model)
    model = MODELS[base_model](config)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if state_dict:
        model.load_state_dict(torch.load(state_dict, weights_only=True), strict=True)
    else:
        print("No state dict provided. Using random initialization.")

    model.train()

    deepspeed.init_distributed()

    pipeline = PipelineModule(
        layers=model.to_layers(),
        loss_fn=loss_fn,
        num_stages=pipeline_stages,
    )
    training_data = PipeDataset(
        dataset_dict["train"],
        tokenizer,
        sequence_length,
        "train",
    )
    engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        config=config_params,
        model=pipeline,
        training_data=training_data,
    )

    if wandb_logging:
        wandb.init(
            config={
                "base_model": base_model,
                "dataset": dataset,
                "dataset_split": dataset_split,
                "pipeline_stages": pipeline_stages,
                "sequence_length": sequence_length,
                "state_dict": state_dict,
                "deepspeed_config": deepspeed_config,
            },
            job_type="training"
        )
        engine.monitor.wandb_monitor = WandbMonitor(
            deepspeed.monitor.config.WandbConfig(
                enabled=False,  # Don't call wandb.init() again
            )
        )
        engine.monitor.wandb_monitor.enabled = True
        engine.monitor.enabled = True

    engine.train()
    for epoch in range(epochs):
        for _ in range(len(training_data) // engine.train_batch_size()):
            loss = engine.train_batch()
            
            if wandb_logging:
                wandb.log(
                    {
                        "epoch": epoch,
                    },
                    step=engine.global_samples,
                )

            if engine.global_steps % validation_stride == 0:
                valid_dataloader = deepspeed.utils.RepeatingLoader(
                    PipeDataset(
                        dataset_dict["test"],
                        tokenizer,
                        sequence_length,
                        "valid",
                    )
                )
                engine.eval_batch(valid_dataloader, compute_loss=True)
                engine.train()

                save_model(engine, output_dir, f"global_step_{engine.global_steps}")
    
    save_model(engine, output_dir, "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--base-model", required=True, type=str, help="Model name")
    parser.add_argument("--deepspeed-config", required=True, type=str, help="Deepspeed config")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--dataset", type=str, help="Dataset name or path")
    parser.add_argument("--dataset-config", type=str, default=None, help="The dataset config on Huggingface")
    parser.add_argument("--dataset-split", type=float, help="Dataset split. If <1, then is a fraction of the dataset")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--pipeline-stages", type=int, default=1, help="Number of pipeline stages")
    parser.add_argument("--sequence-length", type=int, default=8192, help="Token sequence length")
    parser.add_argument("--state_dict", type=str, help="Path to state dict")
    parser.add_argument("--validation-stride", type=int, default=1000, help="Validation stride")
    parser.add_argument("--wandb-logging", action="store_true", help="Log to wandb")
    args = parser.parse_args()

    main(**vars(args))