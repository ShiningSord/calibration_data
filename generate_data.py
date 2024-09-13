from typing import Any, Dict, List
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
import json
# 创建 sampling 参数对象
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, repetition_penalty=1.2, max_tokens=2500, top_k=50)

# 创建一个类用于批量推理
class LLMPredictor:

    def __init__(self):
        # 创建一个 LLM 实例
        self.llm = LLM(model="../Llama-3-8B")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # 生成文本
        outputs = self.llm.generate(batch["prompt"], sampling_params)
        results = []
        for output in outputs:
            for o in output.outputs:
                result = {
                    "prompt": output.prompt,
                    "text": o.text
                }
                results.append(result)
        return results

# 加载数据集
dataset = load_dataset("../cosmopedia-100k", split="train")

# 将数据集分批次进行推理，设置批次大小
batch_size = 32
results = []
predictor = LLMPredictor()
for i in range(50000, 100000, batch_size):
    if i % 500 == 0:
        print(i)
    batch = dataset[i:i + batch_size]
    batch_dict = {"prompt": batch["prompt"]}  # 根据数据集字段名调整
    result = predictor(batch_dict)
    results.extend(result)

# 打印前 10 个结果
# for result in results[:10]:
#     print(f"Prompt: {result['prompt']!r}, Generated text: {result['text']!r}")

# 保存结果到 JSON 文件中
output_filename = "generated_data2.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_filename}")

