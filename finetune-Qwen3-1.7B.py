import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import os
import swanlab
import random


random.seed(42)     # 设置随机种子以确保可重复性
model_dir = "./models"
data_dir = "./datasets"


print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", cache_dir=model_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",cache_dir=model_dir, device_map="auto", torch_dtype=torch.bfloat16)
ds = load_dataset("krisfu/delicate_medical_r1_data_chinese",split="train", cache_dir=data_dir)
print("模型加载完成！")


# ========================== 数据集处理 ==========================
print("正在处理数据集...")

data_list = list(ds)    # 将数据集转换为列表
random.shuffle(data_list)    # 随机打乱数据
split_idx = int(len(data_list) * 0.9)   # 计算分割点

# 分割数据
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

# 保存训练集
with open(os.path.join(data_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# 保存验证集
with open(os.path.join(data_dir, 'val.jsonl'), 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"数据集已分割完成：")
print(f"训练集大小：{len(train_data)}")
print(f"验证集大小：{len(val_data)}")


os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048
swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    })

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []
    # 读取旧的JSONL文件
    with open(origin_path, "r", encoding='utf-8') as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            output = f"<think>{data['think']}</think> \n {data['answer']}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)
    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_func(example):
    """
    将数据集进行预处理
    参数:
        example: 包含输入输出的数据样本字典，应包含'input'和'output'键
    返回值:
        dict: 包含处理后的input_ids、attention_mask和labels的字典
    """ 
    input_ids, attention_mask, labels = [], [], []
    
    # 使用tokenizer对系统指令和用户输入进行编码，不添加特殊token
    # 构造完整的对话格式：系统提示 + 用户输入 + 助手开头
    instruction = tokenizer(
        f"<|system|>\n{PROMPT}<|end|>\n<|user|>\n{example['input']}<|end|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    
    # 对期望的模型输出进行tokenize编码，不添加特殊token
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    # 拼接完整的输入序列：指令tokens + 响应tokens + 填充token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 拼接注意力掩码：指令mask + 响应mask + 1(表示填充位置可见)
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    
    # 构造标签序列：指令部分用-100忽略 + 响应tokens + 填充token
    # -100表示在计算损失时忽略这些位置
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 如果序列长度超过最大长度限制，则进行截断处理
    if len(input_ids) > MAX_LENGTH:  
        # 截取到最大长度
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # 返回处理后的数据字典
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   

def predict(messages, model, tokenizer):
    device = "cuda"
    model.eval()
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
    model.train()
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# Transformers加载模型权重
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

print("正在加载数据集...")

# 加载、处理数据集和测试集
train_dataset_path = "./datasets/train.jsonl"
test_dataset_path = "./datasets/val.jsonl"
train_jsonl_new_path = "./datasets/train_format.jsonl"
test_jsonl_new_path = "./datasets/val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
print(train_ds.column_names)
print(train_ds[0])
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
print(train_dataset.column_names)
print(train_dataset[0])

# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

print("数据集加载完成！")

# print("正在训练模型...")
# args = TrainingArguments(
#     output_dir="./output/Qwen3-1.7B",
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=4,
#     eval_strategy="steps",
#     eval_steps=100,
#     logging_steps=10,
#     num_train_epochs=2,
#     save_steps=400,
#     learning_rate=1e-4,
#     save_on_each_node=True,
#     gradient_checkpointing=True,
#     report_to="swanlab",
#     run_name="qwen3-1.7B",
# )

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
# )

# trainer.train()
# print("训练完成！")

# print("正在生成预测结果...")
# # 用测试集的前3条，主观看模型
# test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
# test_text_list = []
# for index, row in test_df.iterrows():
#     instruction = row['instruction']
#     input_value = row['input']
#     messages = [
#         {"role": "system", "content": f"{instruction}"},
#         {"role": "user", "content": f"{input_value}"}
#     ]
#     response = predict(messages, model, tokenizer)
#     response_text = f"""
#     Question: {input_value}
#     LLM:{response}
#     """

#     test_text_list.append(swanlab.Text(response_text))
#     print(response_text)

# swanlab.log({"Prediction": test_text_list})
# print("预测结果生成完成！")

# swanlab.finish()
