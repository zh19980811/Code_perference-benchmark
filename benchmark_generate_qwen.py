from datasets import load_from_disk, concatenate_datasets
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()
qwen_api_key = os.getenv("qwen")

# Initialize OpenAI-compatible client for Qwen
client_qwen = OpenAI(
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Completion function
def qwen_generate(text):
    completion = client_qwen.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a code generation assistant. Use a function to implement the following code."},
            {"role": "user", "content": text},
        ],
    )
    return completion.choices[0].message.content

# Prompt processing function
def process_prompt(example, idx):
    if example.get("model_answer"):
        return example
    try:
        print(f"[⏳] 处理第 {idx} 条：{example['prompt'][:30]}...")
        model_answer = qwen_generate(example["prompt"])
    except Exception as e:
        print(f"[❌] 第 {idx} 条失败: {e}")
        model_answer = ""
    return {"model_answer": model_answer}

# Load dataset
dataset_path = "qwen_datasets" if os.path.exists("qwen_datasets") else "orgin_datasets"
print(f"加载数据集：{dataset_path}")
dataset = load_from_disk(dataset_path)

# Process train set
train_dataset = dataset["train"]
filtered_dataset = train_dataset.filter(lambda x: x.get("model_answer") is None)
processed_dataset = filtered_dataset.map(process_prompt, with_indices=True)

# Merge processed + unprocessed
final_train = concatenate_datasets([
    train_dataset.filter(lambda x: x.get("model_answer") is not None),
    processed_dataset
])

# Save dataset
dataset["train"] = final_train
dataset.save_to_disk("qwen_datasets")
print("✅ 保存成功！")
