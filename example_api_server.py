"""
示例 API 服务器
用于演示如何创建一个符合 FiNER benchmark 评估要求的 NER API

运行方式:
    # 使用 Flask (简单)
    python example_api_server.py --framework flask

    # 使用 FastAPI (推荐生产环境)
    python example_api_server.py --framework fastapi
"""

import argparse
from typing import List


# 示例1: 使用 Flask
def create_flask_app():
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        预测端点

        输入: {"tokens": ["The", "company", "reported", "revenue"]}
        输出: {"labels": ["O", "O", "O", "B-Revenues"]}
        """
        try:
            data = request.json
            tokens = data.get("tokens", [])

            if not tokens:
                return jsonify({"error": "No tokens provided"}), 400

            # 这里是你的模型预测逻辑
            # 示例：简单的规则基线
            labels = predict_ner(tokens)

            return jsonify({"labels": labels, "num_tokens": len(tokens)})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        """健康检查端点"""
        return jsonify({"status": "healthy"})

    return app


# 示例2: 使用 FastAPI
def create_fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="FiNER NER API", version="1.0.0")

    class PredictionRequest(BaseModel):
        tokens: List[str]

        class Config:
            schema_extra = {
                "example": {
                    "tokens": [
                        "The",
                        "company",
                        "reported",
                        "revenue",
                        "of",
                        "$",
                        "1.2",
                        "million",
                    ]
                }
            }

    class PredictionResponse(BaseModel):
        labels: List[str]
        num_tokens: int

        class Config:
            schema_extra = {
                "example": {
                    "labels": [
                        "O",
                        "O",
                        "O",
                        "B-Revenues",
                        "O",
                        "O",
                        "I-Revenues",
                        "I-Revenues",
                    ],
                    "num_tokens": 8,
                }
            }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        预测端点

        - **tokens**: 输入的 token 列表
        """
        try:
            tokens = request.tokens

            if not tokens:
                raise HTTPException(status_code=400, detail="No tokens provided")

            # 这里是你的模型预测逻辑
            labels = predict_ner(tokens)

            return PredictionResponse(labels=labels, num_tokens=len(tokens))

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """健康检查端点"""
        return {"status": "healthy"}

    return app


def predict_ner(tokens: List[str]) -> List[str]:
    """
    NER 预测函数

    这是一个简单的示例实现。在实际应用中，你应该：
    1. 加载你的预训练模型
    2. 对输入进行预处理
    3. 运行模型推理
    4. 对输出进行后处理

    Args:
        tokens: 输入的 token 列表

    Returns:
        预测的标签列表
    """

    # ============================================================
    # 方法1: 使用 Transformers 模型
    # ============================================================
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch
    
    # 初始化（只需要做一次，可以在全局或类中）
    model_name = "nlpaueb/sec-bert-base"  # 或你自己的模型路径
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 预测
    inputs = tokenizer(
        tokens, 
        is_split_into_words=True, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]
    
    # 处理对齐问题（如果使用 subword tokenization）
    word_ids = inputs.word_ids(batch_index=0)
    aligned_labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            continue
        elif word_idx != previous_word_idx:
            aligned_labels.append(predicted_labels[word_idx])
        previous_word_idx = word_idx
    
    return aligned_labels
    """

    # ============================================================
    # 方法2: 使用 TensorFlow/Keras 模型
    # ============================================================
    """
    import tensorflow as tf
    import numpy as np
    
    # 加载模型
    model = tf.keras.models.load_model("path/to/your/model")
    
    # 预处理
    # ... 你的预处理逻辑 ...
    
    # 预测
    predictions = model.predict(processed_input)
    predicted_labels = np.argmax(predictions, axis=-1)
    
    # 转换为标签字符串
    idx2label = {...}  # 你的标签映射
    labels = [idx2label[idx] for idx in predicted_labels]
    
    return labels
    """

    # ============================================================
    # 方法3: 简单的规则基线（仅用于演示）
    # ============================================================
    labels = []
    for i, token in enumerate(tokens):
        # 简单规则：识别数字和金融关键词
        token_lower = token.lower()

        # 检查是否是数字或货币符号
        if token in ["$", "€", "£", "¥"] or any(c.isdigit() for c in token):
            # 查看前一个 token 是否是金融关键词
            if i > 0:
                prev_token = tokens[i - 1].lower()
                if prev_token in ["revenue", "revenues", "sales"]:
                    labels.append("I-Revenues")
                elif prev_token in ["asset", "assets"]:
                    labels.append("I-Assets")
                elif prev_token in ["liability", "liabilities", "debt"]:
                    labels.append("I-Liabilities")
                else:
                    labels.append("O")
            else:
                labels.append("O")
        # 检查是否是金融关键词
        elif token_lower in ["revenue", "revenues", "sales"]:
            labels.append("B-Revenues")
        elif token_lower in ["asset", "assets"]:
            labels.append("B-Assets")
        elif token_lower in ["liability", "liabilities", "debt"]:
            labels.append("B-Liabilities")
        else:
            labels.append("O")

    return labels


def main():
    parser = argparse.ArgumentParser(description="运行示例 NER API 服务器")
    parser.add_argument(
        "--framework",
        type=str,
        default="flask",
        choices=["flask", "fastapi"],
        help="选择 web 框架",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")

    args = parser.parse_args()

    if args.framework == "flask":
        print(f"Starting Flask server on http://{args.host}:{args.port}")
        app = create_flask_app()
        app.run(host=args.host, port=args.port, debug=False)

    elif args.framework == "fastapi":
        print(f"Starting FastAPI server on http://{args.host}:{args.port}")
        app = create_fastapi_app()

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
