from hendrycks_dataloader import HendrycksDatasetLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import wandb
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--output-dir", type=str, default="output")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

wandb.init(project="Hendrycks-Math-T5-V1_1-Large")

MODEL_NAME = "google/t5-v1_1-large"

print("Loading dataset...")
dataloader = HendrycksDatasetLoader()
datasets = dataloader.load_from_source()
train_dataset = datasets["train"]
eval_dataset = datasets["test"].select(range(800))

print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

max_input_length = 192
max_output_length = 512


def preprocess_function(examples):
    inputs = examples["input"]
    targets = [
        process + "\nAnswer: " + label
        for process, label in zip(examples["process"], examples["label"])
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def extract_label(s):
        s = s.lower()
        if "answer:" in s:
            return s.split("answer:")[-1].strip()
        try:
            return float(s.strip())
        except:
            return s.strip()

    pred_label = [extract_label(p) for p in pred_str]
    true_label = [extract_label(l) for l in label_str]

    correct = [p == l for p, l in zip(pred_label, true_label)]
    acc = sum(correct) / len(correct)
    return {"accuracy": acc}


training_args = TrainingArguments(
    output_dir=args.output_dir,
    run_name="t5-large-hendrycks-math",
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    weight_decay=0.01,
    num_train_epochs=args.epochs,
    save_steps=100,
    save_total_limit=2,
    fp16=False,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("Start training...")

if args.resume:
    checkpoint_files = [
        f for f in os.listdir(args.output_dir)
        if re.match(r"^checkpoint-\d+$", f)
    ]
    max_epoch_ckpt = max(checkpoint_files, key=lambda x: int(x.split('-')[1]))
    checkpoint_path = os.path.join(args.output_dir, max_epoch_ckpt)
    print(f"Resuming from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()

print("Training finished.")
