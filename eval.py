import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

random.seed(42)     # 设置随机种子以确保可重复性
model_dir = "./models"

model_dir_origin = "./models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
model_dir_finetuned = "./outputs/checkpoint-1122"

load_original_model = False    # 是否加载原始模型

print(f"正在加载{"原始" if load_original_model else "微调后"}的模型权重...")

if load_original_model:
    tokenizer = AutoTokenizer.from_pretrained(model_dir_origin)
    model = AutoModelForCausalLM.from_pretrained(model_dir_origin, device_map="auto", torch_dtype=torch.bfloat16)

else:
    tokenizer = AutoTokenizer.from_pretrained(model_dir_origin)
    model = AutoModelForCausalLM.from_pretrained(model_dir_finetuned, device_map="auto", torch_dtype=torch.bfloat16)

print("模型加载完成！")


PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 加载、处理数据集和测试集
train_dataset_path = "./datasets/train.jsonl"
test_dataset_path = "./datasets/val.jsonl"
train_jsonl_new_path = "./datasets/train_format.jsonl"
test_jsonl_new_path = "./datasets/val_format.jsonl"


print("正在生成预测结果...")
# 用测试集的前3条，主观看模型
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]
    response = predict(messages, model, tokenizer)
    response_text = f"""
    Question: {input_value}
    LLM:{response}
    """

    print(response_text)

print("预测结果生成完成！")
