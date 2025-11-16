# FiNER-139 API 评估指南

本指南介绍如何使用 FiNER-139 benchmark 来评估你自己的命名实体识别（NER）API。

## 目录

1. [快速开始](#快速开始)
2. [API 接口要求](#api-接口要求)
3. [评估方式](#评估方式)
4. [示例代码](#示例代码)
5. [评估指标说明](#评估指标说明)

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 测试评估脚本

运行演示模式来测试脚本是否正常工作：

```bash
python evaluate_custom_api.py --demo --use_local_function
```

### 3. 评估你的 API

```bash
# 评估远程 API
python evaluate_custom_api.py \
    --api_endpoint "http://your-api-url/predict" \
    --split "test"

# 或者评估本地模型
python evaluate_custom_api.py \
    --use_local_function \
    --split "test"
```

---

## API 接口要求

### 输入格式

你的 API 应该接受以下 JSON 格式的 POST 请求：

```json
{
    "tokens": ["The", "company", "reported", "revenue", "of", "$", "1.2", "million"]
}
```

### 输出格式

你的 API 应该返回以下格式的响应：

```json
{
    "labels": ["O", "O", "O", "B-Revenues", "O", "O", "I-Revenues", "I-Revenues"]
}
```

或者：

```json
{
    "predictions": ["O", "O", "O", "B-Revenues", "O", "O", "I-Revenues", "I-Revenues"]
}
```

### 标签格式

- 使用 **IOB2** 标注格式
- `B-` 开头表示实体的开始（Begin）
- `I-` 开头表示实体的内部（Inside）
- `O` 表示非实体（Outside）
- FiNER-139 使用 139 个实体类型，例如：`Revenues`, `Assets`, `Liabilities` 等

---

## 评估方式

### 方式一：评估远程 API

如果你的模型部署为在线服务：

```bash
python evaluate_custom_api.py \
    --api_endpoint "http://your-api-url/predict" \
    --api_key "your-api-key" \
    --split "test" \
    --output "results.json"
```

### 方式二：评估本地模型

修改 `evaluate_custom_api.py` 中的 `predict_local` 方法，实现你的本地预测逻辑：

```python
def predict_local(self, tokens: List[str]) -> List[str]:
    # 加载你的模型
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    
    model = AutoModelForTokenClassification.from_pretrained("your_model_path")
    tokenizer = AutoTokenizer.from_pretrained("your_model_path")
    
    # 进行预测
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)
    
    # 将预测索引转换为标签
    predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
    
    return predicted_labels
```

然后运行：

```bash
python evaluate_custom_api.py --use_local_function --split "test"
```

### 方式三：快速测试（限制样本数）

用少量样本进行快速测试：

```bash
python evaluate_custom_api.py \
    --api_endpoint "http://your-api-url/predict" \
    --split "validation" \
    --max_samples 100
```

---

## 示例代码

### 创建一个简单的 API 服务器（使用 Flask）

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

app = Flask(__name__)

# 加载你的模型
model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tokens = data.get('tokens', [])
    
    # 进行预测
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", 
                      padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = outputs.logits.argmax(-1)
    labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
    
    return jsonify({
        "labels": labels
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### 创建一个 FastAPI 服务器

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

app = FastAPI()

# 加载模型
model = AutoModelForTokenClassification.from_pretrained("nlpaueb/sec-bert-base")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")

class PredictionRequest(BaseModel):
    tokens: List[str]

class PredictionResponse(BaseModel):
    labels: List[str]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    tokens = request.tokens
    
    # 进行预测
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                      padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = outputs.logits.argmax(-1)
    labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
    
    return PredictionResponse(labels=labels)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

### 测试你的 API

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "tokens": ["The", "company", "reported", "revenue", "of", "$", "1.2", "million"]
    }
)

print(response.json())
# 输出: {"labels": ["O", "O", "O", "B-Revenues", "O", "O", "I-Revenues", "I-Revenues"]}
```

---

## 评估指标说明

评估脚本会输出以下指标：

### 1. 实体级别的指标

- **Precision（精确率）**：预测为正例的样本中真正例的比例
- **Recall（召回率）**：真正例中被正确预测的比例
- **F1-Score**：精确率和召回率的调和平均数

### 2. 宏观和微观平均

- **Macro F1**：每个类别的 F1 分数的平均值（不考虑类别不平衡）
- **Micro F1**：所有类别的 TP、FP、FN 合并计算的 F1 分数（考虑类别不平衡）
- **Weighted F1**：按每个类别的样本数量加权的 F1 平均值

### 3. 输出示例

```
              precision    recall  f1-score   support

    Revenues     0.8542    0.7823    0.8167       845
      Assets     0.9012    0.8534    0.8766      1203
 Liabilities     0.7834    0.8001    0.7916       542
         ...       ...       ...       ...       ...

   micro avg     0.8534    0.8201    0.8364     15234
   macro avg     0.8123    0.7956    0.8035     15234
weighted avg     0.8556    0.8201    0.8371     15234
```

---

## 数据集分割

FiNER-139 包含三个数据集分割：

- **train**: 训练集（用于模型训练）
- **validation**: 验证集（用于模型调优）
- **test**: 测试集（用于最终评估）

建议在 `test` 分割上进行最终评估。

---

## 高级用法

### 1. 自定义评估函数

如果你需要自定义评估逻辑，可以直接使用 `CustomAPIEvaluator` 类：

```python
from evaluate_custom_api import CustomAPIEvaluator

evaluator = CustomAPIEvaluator(
    api_endpoint="http://your-api-url/predict",
    api_key="your-api-key"
)

# 评估单个样本
result = evaluator.evaluate_sample(
    tokens=["The", "revenue", "was", "$", "1.2", "million"],
    true_labels=["O", "B-Revenues", "O", "O", "I-Revenues", "I-Revenues"]
)
print(result)

# 评估整个数据集
results = evaluator.evaluate(split="test")
```

### 2. 批量处理

如果你的 API 支持批量处理，可以修改 `call_api` 方法来提高效率：

```python
def call_api_batch(self, batch_tokens: List[List[str]]) -> List[List[str]]:
    """批量调用 API"""
    payload = {"batch_tokens": batch_tokens}
    response = requests.post(self.api_endpoint, json=payload)
    return response.json()["batch_labels"]
```

### 3. 错误分析

保存预测结果以进行错误分析：

```python
import json

# 在 evaluate 方法中保存详细结果
detailed_results = []
for idx, (tokens, true_labels, pred_labels) in enumerate(zip(all_tokens, y_true_all, y_pred_all)):
    detailed_results.append({
        "id": idx,
        "tokens": tokens,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "errors": [i for i, (t, p) in enumerate(zip(true_labels, pred_labels)) if t != p]
    })

with open("detailed_results.json", "w") as f:
    json.dump(detailed_results, f, indent=2)
```

---

## 常见问题

### Q1: 我的 API 返回的标签数量与输入 token 数量不一致怎么办？

A: 脚本会自动处理这种情况，通过填充 `O` 标签或截断来匹配长度，但会输出警告。建议修复 API 以确保输出长度匹配。

### Q2: 如何处理 subword tokenization？

A: FiNER-139 使用 word-level tokens。如果你的模型使用 subword tokenization，需要在 API 中实现 subword 到 word 的对齐。

### Q3: 评估速度太慢怎么办？

A: 可以：
1. 使用 `--max_samples` 限制评估样本数
2. 实现批量预测
3. 使用更小的数据集分割（validation 比 test 小）

### Q4: 如何获取数据集的标签列表？

A: 运行以下代码：

```python
import datasets
dataset = datasets.load_dataset('nlpaueb/finer-139', split='train', streaming=True)
labels = dataset.features['ner_tags'].feature.names
print(f"Total labels: {len(labels)}")
print(labels[:10])  # 打印前 10 个标签
```

---

## 参考资料

- [FiNER-139 论文](https://arxiv.org/abs/2203.06482)
- [FiNER-139 数据集](https://huggingface.co/datasets/nlpaueb/finer-139)
- [SEC-BERT 模型](https://huggingface.co/nlpaueb/sec-bert-base)
- [seqeval 文档](https://github.com/chakki-works/seqeval)

---

## 支持

如有问题，请查看：
- 原始 README.md
- 论文和数据集文档
- GitHub Issues

