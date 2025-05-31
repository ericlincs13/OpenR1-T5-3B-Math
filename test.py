from datasets import load_dataset
from transformers import T5Tokenizer
import numpy as np

dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-large")

input_lens = []
output_lens = []

for example in dataset:
    input_ids = tokenizer.encode(example["problem"], truncation=False)
    output_text = example["solution"] + "\nAnswer: " + example["answer"]
    output_ids = tokenizer.encode(output_text, truncation=False)
    input_lens.append(len(input_ids))
    output_lens.append(len(output_ids))

print(f"Input length - mean: {np.mean(input_lens):.1f}, median: {np.median(input_lens)}, 90% percentile: {np.percentile(input_lens, 90)}")
print(f"Output length - mean: {np.mean(output_lens):.1f}, median: {np.median(output_lens)}, 90% percentile: {np.percentile(output_lens, 90)}")
