from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=bool, default=True)
args = parser.parse_args()

wandb.init(project="OpenR1-T5-Large-Math")

MODEL_NAME = "google-t5/t5-large"

print("Loading dataset...")
dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

print("Creating train and eval splits...")
split = dataset.train_test_split(test_size=0.05, seed=42)  # type: ignore
train_dataset = split["train"]
eval_dataset = split["test"]

print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

max_input_length = 1024
max_output_length = 2800


def preprocess_function(examples):
    inputs = examples["problem"]
    targets = [
        s + "\nAnswer: " + a
        for s, a in zip(examples["solution"], examples["answer"])
    ]
    model_inputs = tokenizer(inputs,
                             max_length=max_input_length,
                             truncation=True,
                             padding="max_length")

    labels = tokenizer(text_target=targets,
                       max_length=max_output_length,
                       truncation=True,
                       padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100)
                               for l in label]
                              for label in model_inputs["labels"]]
    return model_inputs


print("Tokenizing train dataset...")
train_dataset = train_dataset.map(preprocess_function,
                                  batched=True,
                                  remove_columns=train_dataset.column_names,
                                  load_from_cache_file=args.cache)

print("Tokenizing eval dataset...")
eval_dataset = eval_dataset.map(preprocess_function,
                                batched=True,
                                remove_columns=eval_dataset.column_names,
                                load_from_cache_file=args.cache)

training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    fp16=False,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Start training...")
trainer.train()

print("Training finished.")
