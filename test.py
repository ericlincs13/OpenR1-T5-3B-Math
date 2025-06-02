from transformers import T5Tokenizer
import numpy as np
from svamp_dataloader import SVAMPDatasetLoader

MODEL_NAME = "google-t5/t5-large"

svamp_loader = SVAMPDatasetLoader()
datasets = svamp_loader.load_from_source()
train_dataset = datasets["train"]
eval_dataset = datasets["test"]

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

input_lens = []
output_lens = []

for example in train_dataset:
    input_ids = tokenizer.encode(example["input"], truncation=False)
    output_text = example["label"]
    output_ids = tokenizer.encode(output_text, truncation=False)
    input_lens.append(len(input_ids))
    output_lens.append(len(output_ids))

print(
    f"Input length - mean: {np.mean(input_lens):.1f}, median: {np.median(input_lens)}, 90% percentile: {np.percentile(input_lens, 90)}"
)
print(
    f"Output length - mean: {np.mean(output_lens):.1f}, median: {np.median(output_lens)}, 90% percentile: {np.percentile(output_lens, 90)}"
)
