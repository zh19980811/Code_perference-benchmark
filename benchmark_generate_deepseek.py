from datasets import load_from_disk
from dotenv import load_dotenv
from openai import OpenAI
import os
from openai import OpenAI
import time

load_dotenv()
deepseek_api_key =os.getenv("deepseek_api")
client_deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")



client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def deepseek(text):
    completion = client.chat.completions.create(
    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=[
        {'role': 'user', 'content': '9.9和9.11谁大'}
    ]
    )
    return response.choices[0].message.content



def process_prompt(example,idx):
    if example.get("model_answer"):
        return example
    try:
        print(f"处理第 {idx} 条：{example['prompt'][:30]}...")
        model_answer=deepseek(example["prompt"])
    except Exception as e:
        print(f"处理第 {idx} 条失败: {e}")
        model_answer = ""
    return {"model_answer": model_answer}

    
if os.path.exists("deepseek_datasets"):
    print("加载已保存的数据...")
    dataset = load_from_disk("deepseek_datasets")
else:
    print("加载原始数据...")
    dataset = load_from_disk("orgin_datasets")

# 处理 train 数据集
train_dataset = dataset["train"]

# 仅处理还没有 model_answer 的
filtered_dataset = train_dataset.filter(lambda x: x.get("model_answer") is None)

# 加上 with_indices=True
processed_dataset = filtered_dataset.map(process_prompt, with_indices=True)

# 合并已处理 + 新处理的数据
from datasets import concatenate_datasets
final_train = concatenate_datasets([
    train_dataset.filter(lambda x: x.get("model_answer") is not None),
    processed_dataset
])

# 更新并保存
dataset["train"] = final_train
dataset.save_to_disk("deepseek_datasets")
print("保存成功 ✅")