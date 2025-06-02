from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import argparse
from hendrycks_dataloader import HendrycksDatasetLoader

MODEL_NAME = "google/t5-v1_1-large"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(args.checkpoint,
                                                   device_map=device)
model.eval()

max_input_length = 192
max_output_length = 512

print("Loading dataset...")
dataloader = HendrycksDatasetLoader()
datasets = dataloader.load_from_source()
dataset = datasets["test"]

correct = 0
total = 0


def extract_answer(s):
    s = s.lower()
    if "answer:" in s:
        return s.split("answer:")[-1].strip()
    try:
        return float(s.strip())
    except:
        return s.strip()


for example in dataset:
    input_text = example["input"]  # type: ignore
    answer_text = example["label"]  # type: ignore
    inputs = tokenizer(input_text,
                       max_length=max_input_length,
                       truncation=True,
                       padding="max_length",
                       return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 max_length=max_output_length,
                                 num_beams=2,
                                 early_stopping=True)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_answer = extract_answer(pred)
    truth_answer = extract_answer(answer_text)

    print("==============================")
    print("題目:", input_text)
    print("預測答案:", pred_answer)
    print("正確答案:", truth_answer)

    if pred_answer == truth_answer:
        correct += 1
    total += 1

print(f"\n總共 {total} 題，正確 {correct} 題，正確率：{correct/total:.2%}")
