# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""
Trillim LoRA Finetuner — local training script.

Usage:
    uv run finetune/train.py
    uv run finetune/train.py --config finetune/my_config.json

All hyperparameters can be set via the CONFIG dict below, overridden with a
JSON config file (--config), or passed as CLI flags (--lr, --epochs, etc.).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Default hyperparameters — edit here or override via CLI / JSON config
# ---------------------------------------------------------------------------
CONFIG = {
    # Model
    "model_id": "microsoft/bitnet-b1.58-2B-4T-bf16",

    # Dataset
    "dataset_dir": os.path.join(os.path.dirname(__file__), "dataset"),
    "dataset_format": "text",  # "text", "json", or "csv"

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["k_proj", "q_proj", "v_proj", "o_proj"],

    # Training
    "max_seq_length": 256,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "num_epochs": 5,
    "warmup_steps": 100,
    "logging_steps": 10,
    "optimizer": "paged_adamw_8bit",
    "bf16": True,

    # Checkpointing
    "checkpoint_every_n_epochs": 1,  # save a checkpoint every N epochs
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "checkpoints"),
    "output_dir": os.path.join(os.path.dirname(__file__), "finetuned_model"),

    # Chat template written into the saved tokenizer so model inference
    # matches the training format.  Set to "" to keep the model's default.
    "chat_template": (
        "{% for message in messages %}"
        "{{ message['role'] | capitalize }}: {{ message['content'] | trim }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
    ),

    # Resume from a checkpoint (path or ""). Leave empty to start fresh.
    "resume_from": "",
}


# ---------------------------------------------------------------------------
# Epoch-based checkpoint callback
# ---------------------------------------------------------------------------
class CheckpointEveryNEpochs(TrainerCallback):
    """Save LoRA adapter + tokenizer every *n* completed epochs."""

    def __init__(self, n: int, checkpoint_dir: str, tokenizer):
        self.n = max(n, 1)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        if epoch % self.n == 0:
            save_path = self.checkpoint_dir / f"checkpoint-epoch-{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            print(f"\n>> Checkpoint saved: {save_path}\n")


# ---------------------------------------------------------------------------
# Dataset loading  (edit this function for custom formats)
# ---------------------------------------------------------------------------
def load_training_dataset(cfg: dict):
    """Load and preprocess the training dataset.

    Modify this function to handle your own dataset format.  The only
    requirement is that the returned dataset has a ``text`` column containing
    the fully-formatted training strings.
    """
    dataset_dir = cfg["dataset_dir"]
    fmt = cfg["dataset_format"]

    if fmt == "text":
        files = sorted(Path(dataset_dir).glob("*.txt"))
        if not files:
            sys.exit(f"No .txt files found in {dataset_dir}")
        dataset = load_dataset("text", data_files={"train": [str(f) for f in files]}, split="train")

        def format_text(example):
            example["text"] = example["text"].replace("\\n", "\n")
            return example

        dataset = dataset.map(format_text)

    elif fmt == "json":
        files = sorted(Path(dataset_dir).glob("*.json")) + sorted(Path(dataset_dir).glob("*.jsonl"))
        if not files:
            sys.exit(f"No .json/.jsonl files found in {dataset_dir}")
        dataset = load_dataset("json", data_files={"train": [str(f) for f in files]}, split="train")

    elif fmt == "csv":
        files = sorted(Path(dataset_dir).glob("*.csv"))
        if not files:
            sys.exit(f"No .csv files found in {dataset_dir}")
        dataset = load_dataset("csv", data_files={"train": [str(f) for f in files]}, split="train")

    else:
        sys.exit(f"Unknown dataset_format: {fmt!r}")

    print(f"Loaded {len(dataset)} training examples")
    print(f"First example: {repr(dataset[0]['text'][:200])}")
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Trillim LoRA finetuner")
    parser.add_argument("--config", type=str, default="", help="Path to JSON config file")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-format", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--epochs", type=int, default=None, dest="num_epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None, dest="checkpoint_every_n_epochs")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    # Build effective config: defaults <- JSON file <- CLI flags
    cfg = dict(CONFIG)

    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    for key in [
        "model_id", "dataset_dir", "dataset_format", "learning_rate",
        "num_epochs", "batch_size", "checkpoint_every_n_epochs", "resume_from",
    ]:
        cli_val = getattr(args, key.replace("-", "_"), None)
        if cli_val is not None:
            cfg[key] = cli_val

    print("=" * 60)
    print("Trillim LoRA Finetuner")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # ---- Tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])

    # ---- Dataset ----------------------------------------------------------
    dataset = load_training_dataset(cfg)

    # ---- Model ------------------------------------------------------------
    print(f"Loading model: {cfg['model_id']} ...")

    try:
        from transformers import BitNetQuantConfig
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            quantization_config=BitNetQuantConfig(),
        )
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model = prepare_model_for_kbit_training(model)

    # ---- Resume from checkpoint -------------------------------------------
    resume_from = cfg.get("resume_from", "")
    if resume_from:
        from peft import PeftModel
        print(f"Resuming from checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        peft_config = None  # already applied
    else:
        peft_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg["target_modules"],
        )

    # ---- Trainer ----------------------------------------------------------
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    training_args = SFTConfig(
        output_dir=cfg["checkpoint_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy="no",  # we handle saves via the callback
        report_to="none",
        optim=cfg["optimizer"],
        max_seq_length=cfg["max_seq_length"],
    )

    checkpoint_cb = CheckpointEveryNEpochs(
        n=cfg["checkpoint_every_n_epochs"],
        checkpoint_dir=cfg["checkpoint_dir"],
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[checkpoint_cb],
    )

    # ---- Train ------------------------------------------------------------
    print("\nStarting finetuning ...\n")
    trainer.train()

    # ---- Save final model -------------------------------------------------
    trainer.model.save_pretrained(cfg["output_dir"])

    if cfg["chat_template"]:
        tokenizer.chat_template = cfg["chat_template"]
    tokenizer.save_pretrained(cfg["output_dir"])

    print(f"\nFinetuning complete.")
    print(f"  Final adapter saved to: {cfg['output_dir']}")
    print(f"  Checkpoints in:         {cfg['checkpoint_dir']}")


if __name__ == "__main__":
    main()
