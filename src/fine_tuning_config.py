"""
Fine-Tuning Configuration for the Movie Safety Classifier.
Defines all hyperparameters, model choices, and training settings for LoRA/QLoRA.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os

from config import DATA_PATH


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    r: int = 16                              # LoRA rank
    lora_alpha: int = 16                     # LoRA scaling factor
    lora_dropout: float = 0.05               # Dropout for LoRA layers
    bias: str = "none"                       # Bias type ("none", "all", "lora_only")
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"       # MLP layers
    ])
    use_gradient_checkpointing: bool = True   # Save memory
    use_rslora: bool = False                  # Rank-stabilized LoRA
    loftq_config: Optional[dict] = None       # LoftQ quantization


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True                # QLoRA

    # Training
    output_dir: str = "./fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 0.3
    seed: int = 42

    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 50

    # Dataset
    train_file: str = field(default_factory=lambda: os.path.join(DATA_PATH, "preference_pairs.jsonl"))
    test_size: float = 0.2

    def to_dict(self) -> dict:
        """Convert to dictionary for HuggingFace Trainer."""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "optim": self.optim,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "report_to": "none",  # Set to "wandb" for experiment tracking
        }


@dataclass
class FineTuningConfig:
    """Master configuration for fine-tuning."""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    method: str = "dpo"  # "sft" or "dpo"

    def __post_init__(self):
        """Validate configuration."""
        assert self.method in ["sft", "dpo"], "method must be 'sft' or 'dpo'"
        assert self.lora.r > 0, "LoRA rank must be > 0"
        assert self.training.num_train_epochs > 0, "Must train for at least 1 epoch"

    def summary(self) -> str:
        """Human-readable summary of the config."""
        return f"""
        ╔══════════════════════════════════════════════════════════╗
        ║            FINE-TUNING CONFIGURATION                     ║
        ╠══════════════════════════════════════════════════════════╣
        ║ Method:          {self.method.upper():<40} ║
        ║ Model:           {self.training.model_name:<40} ║
        ║ Max Seq Length:  {self.training.max_seq_length:<40} ║
        ║ Load in 4-bit:   {str(self.training.load_in_4bit):<40} ║
        ║                                                          ║
        ║ LoRA Rank (r):   {self.lora.r:<40} ║
        ║ LoRA Alpha:      {self.lora.lora_alpha:<40} ║
        ║ LoRA Dropout:    {self.lora.lora_dropout:<40} ║
        ║                                                          ║
        ║ Epochs:          {self.training.num_train_epochs:<40} ║
        ║ Batch Size:      {self.training.per_device_train_batch_size:<40} ║
        ║ Grad Accum:      {self.training.gradient_accumulation_steps:<40} ║
        ║ Learning Rate:   {self.training.learning_rate:<40} ║
        ║                                                          ║
        ║ Dataset:         {self.training.train_file:<40} ║
        ║                                                          ║
        ╚══════════════════════════════════════════════════════════╝
        """


# Preset configurations for common scenarios
PRESETS = {
    "quick_test": FineTuningConfig(
        method="sft",
        lora=LoRAConfig(r=8, lora_alpha=8),
        training=TrainingConfig(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_seq_length=512,
        )
    ),
    "standard": FineTuningConfig(
        method="sft",
        lora=LoRAConfig(r=16, lora_alpha=16),
        training=TrainingConfig(
            num_train_epochs=3,
            per_device_train_batch_size=2,
            max_seq_length=2048,
        )
    ),
    "dpo_preference": FineTuningConfig(
        method="dpo",
        lora=LoRAConfig(r=16, lora_alpha=16),
        training=TrainingConfig(
            num_train_epochs=2,
            per_device_train_batch_size=1,
            max_seq_length=1024,
            learning_rate=5e-5,  # Lower LR for DPO
        )
    ),
}


def main():
    """Print all preset configurations."""
    for name, config in PRESETS.items():
        print(f"\n📋 Preset: {name}")
        print(config.summary())


if __name__ == "__main__":
    main()