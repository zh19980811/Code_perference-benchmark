from datasets import load_from_disk
from dotenv import load_dotenv
from openai import OpenAI
import os
import time

load_dotenv()
deepseek_api_key =os.getenv("doubao")
client_deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=deepseek_api_key,
)


def deepseek(text):
      response = client.chat.completions.create(
          model="doubao-1-5-pro-256k-250115",
          messages=[
            {"role":"system","content":"You are a code generation assistant,Use a function to implement the following code"},
            {"role":"user","content":text},
          ],
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

    
if os.path.exists("mis_datasets"):
    print("加载已保存的数据...")
    dataset = load_from_disk("mis_datasets")
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
dataset.save_to_disk("doubao_datasets")
print("保存成功 ✅")