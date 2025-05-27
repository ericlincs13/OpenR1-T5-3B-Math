from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import wandb

wandb.init(project="OpenR1-T5-3B-Math")

MODEL_NAME = "google-t5/t5-3b"

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
    targets = examples["answer"]
    model_inputs = tokenizer(inputs,
                             max_length=max_input_length,
                             truncation=True,
                             padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets,
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
                                  remove_columns=train_dataset.column_names)

print("Tokenizing eval dataset...")
eval_dataset = eval_dataset.map(preprocess_function,
                                batched=True,
                                remove_columns=eval_dataset.column_names)

training_args = TrainingArguments(
    output_dir="./t5-3b-finetuned-openr1",
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
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
