#!/usr/bin/env python3
"""
可扩展的 LLM 评估脚本 - FiNER-139 Benchmark

核心设计：
- call_api() 函数是开放接口，可以重写来适配不同的 LLM
- 支持自定义 prompt 和输出解析逻辑
- 提供完整的评估流程

使用方法:
1. 继承 FiNERLLMEvaluator 类并重写 call_api 方法
2. 或者直接修改 call_api 方法来适配你的 LLM API

运行:
    python evaluate_llm.py --max_samples 10  # 快速测试
    python evaluate_llm.py --split test       # 完整评估
"""

import argparse
import datasets
import itertools
import json
import logging
import os
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


class FiNERLLMEvaluator:
    """
    FiNER-139 LLM 评估器
    
    核心方法：
    - call_api(): 需要重写的接口，用于调用你的 LLM
    - evaluate(): 在数据集上运行完整评估
    """

    def __init__(self):
        """初始化评估器"""
        self.tag2idx, self.idx2tag = self.load_dataset_tags()
        
    @staticmethod
    def load_dataset_tags() -> Tuple[Dict, Dict]:
        """加载数据集标签"""
        # FiNER-139 的固定标签列表
        dataset_tags = [
            "O",
            "B-AccountsPayableCurrent",
            "I-AccountsPayableCurrent",
            "B-AccountsReceivableNetCurrent",
            "I-AccountsReceivableNetCurrent",
            "B-AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
            "I-AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
            "B-AdditionalPaidInCapital",
            "I-AdditionalPaidInCapital",
            "B-AdditionalPaidInCapitalCommonStock",
            "I-AdditionalPaidInCapitalCommonStock",
            "B-AllocatedShareBasedCompensationExpense",
            "I-AllocatedShareBasedCompensationExpense",
            "B-AllowanceForDoubtfulAccountsReceivableCurrent",
            "I-AllowanceForDoubtfulAccountsReceivableCurrent",
            "B-AmortizationOfIntangibleAssets",
            "I-AmortizationOfIntangibleAssets",
            "B-Assets",
            "I-Assets",
            "B-AssetsCurrent",
            "I-AssetsCurrent",
            "B-AssetsNoncurrent",
            "I-AssetsNoncurrent",
            "B-AvailableForSaleSecuritiesDebtSecuritiesCurrent",
            "I-AvailableForSaleSecuritiesDebtSecuritiesCurrent",
            "B-Cash",
            "I-Cash",
            "B-CashAndCashEquivalentsAtCarryingValue",
            "I-CashAndCashEquivalentsAtCarryingValue",
            "B-CashAndCashEquivalentsPeriodIncreaseDecrease",
            "I-CashAndCashEquivalentsPeriodIncreaseDecrease",
            "B-CashCashEquivalentsAndShortTermInvestments",
            "I-CashCashEquivalentsAndShortTermInvestments",
            "B-CommonStockSharesAuthorized",
            "I-CommonStockSharesAuthorized",
            "B-CommonStockSharesIssued",
            "I-CommonStockSharesIssued",
            "B-CommonStockSharesOutstanding",
            "I-CommonStockSharesOutstanding",
            "B-CommonStockValue",
            "I-CommonStockValue",
            "B-ComprehensiveIncomeNetOfTax",
            "I-ComprehensiveIncomeNetOfTax",
            "B-CostOfRevenue",
            "I-CostOfRevenue",
            "B-DeferredIncomeTaxExpenseBenefit",
            "I-DeferredIncomeTaxExpenseBenefit",
            "B-DeferredIncomeTaxLiabilities",
            "I-DeferredIncomeTaxLiabilities",
            "B-DeferredRevenueCurrent",
            "I-DeferredRevenueCurrent",
            "B-Depreciation",
            "I-Depreciation",
            "B-DepreciationDepletionAndAmortization",
            "I-DepreciationDepletionAndAmortization",
            "B-EarningsPerShareBasic",
            "I-EarningsPerShareBasic",
            "B-EarningsPerShareDiluted",
            "I-EarningsPerShareDiluted",
            "B-EffectiveIncomeTaxRateContinuingOperations",
            "I-EffectiveIncomeTaxRateContinuingOperations",
            "B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized",
            "I-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized",
            "B-EquityComponentDomain",
            "I-EquityComponentDomain",
            "B-FiniteLivedIntangibleAssetsAccumulatedAmortization",
            "I-FiniteLivedIntangibleAssetsAccumulatedAmortization",
            "B-FiniteLivedIntangibleAssetsAmortizationExpenseNextTwelveMonths",
            "I-FiniteLivedIntangibleAssetsAmortizationExpenseNextTwelveMonths",
            "B-FiniteLivedIntangibleAssetsByMajorClassAxis",
            "I-FiniteLivedIntangibleAssetsByMajorClassAxis",
            "B-FiniteLivedIntangibleAssetsGross",
            "I-FiniteLivedIntangibleAssetsGross",
            "B-FiniteLivedIntangibleAssetsNet",
            "I-FiniteLivedIntangibleAssetsNet",
            "B-Goodwill",
            "I-Goodwill",
            "B-GrossProfit",
            "I-GrossProfit",
            "B-IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "I-IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "B-IncomeTaxExpenseBenefit",
            "I-IncomeTaxExpenseBenefit",
            "B-IncomeTaxesPaidNet",
            "I-IncomeTaxesPaidNet",
            "B-IncreaseDecreaseInAccountsPayable",
            "I-IncreaseDecreaseInAccountsPayable",
            "B-IncreaseDecreaseInAccountsReceivable",
            "I-IncreaseDecreaseInAccountsReceivable",
            "B-IncreaseDecreaseInDeferredRevenue",
            "I-IncreaseDecreaseInDeferredRevenue",
            "B-IncreaseDecreaseInInventories",
            "I-IncreaseDecreaseInInventories",
            "B-IncreaseDecreaseInOperatingCapital",
            "I-IncreaseDecreaseInOperatingCapital",
            "B-IncreaseDecreaseInStockholdersEquity",
            "I-IncreaseDecreaseInStockholdersEquity",
            "B-InterestExpense",
            "I-InterestExpense",
            "B-InterestPaidNet",
            "I-InterestPaidNet",
            "B-InventoryNet",
            "I-InventoryNet",
            "B-Liabilities",
            "I-Liabilities",
            "B-LiabilitiesAndStockholdersEquity",
            "I-LiabilitiesAndStockholdersEquity",
            "B-LiabilitiesCurrent",
            "I-LiabilitiesCurrent",
            "B-LiabilitiesNoncurrent",
            "I-LiabilitiesNoncurrent",
            "B-LongTermDebt",
            "I-LongTermDebt",
            "B-LongTermDebtCurrent",
            "I-LongTermDebtCurrent",
            "B-LongTermDebtNoncurrent",
            "I-LongTermDebtNoncurrent",
            "B-NetCashProvidedByUsedInFinancingActivities",
            "I-NetCashProvidedByUsedInFinancingActivities",
            "B-NetCashProvidedByUsedInInvestingActivities",
            "I-NetCashProvidedByUsedInInvestingActivities",
            "B-NetCashProvidedByUsedInOperatingActivities",
            "I-NetCashProvidedByUsedInOperatingActivities",
            "B-NetIncomeLoss",
            "I-NetIncomeLoss",
            "B-NumberOfOperatingSegments",
            "I-NumberOfOperatingSegments",
            "B-OperatingExpenses",
            "I-OperatingExpenses",
            "B-OperatingIncomeLoss",
            "I-OperatingIncomeLoss",
            "B-OperatingLeaseLiabilityCurrent",
            "I-OperatingLeaseLiabilityCurrent",
            "B-OperatingLeaseRightOfUseAsset",
            "I-OperatingLeaseRightOfUseAsset",
            "B-OtherComprehensiveIncomeLossNetOfTax",
            "I-OtherComprehensiveIncomeLossNetOfTax",
            "B-PaymentsForRepurchaseOfCommonStock",
            "I-PaymentsForRepurchaseOfCommonStock",
            "B-PaymentsToAcquirePropertyPlantAndEquipment",
            "I-PaymentsToAcquirePropertyPlantAndEquipment",
            "B-PreferredStockSharesAuthorized",
            "I-PreferredStockSharesAuthorized",
            "B-PreferredStockSharesIssued",
            "I-PreferredStockSharesIssued",
            "B-PreferredStockSharesOutstanding",
            "I-PreferredStockSharesOutstanding",
            "B-PrepaidExpenseAndOtherAssetsCurrent",
            "I-PrepaidExpenseAndOtherAssetsCurrent",
            "B-PropertyPlantAndEquipmentNet",
            "I-PropertyPlantAndEquipmentNet",
            "B-ResearchAndDevelopmentExpense",
            "I-ResearchAndDevelopmentExpense",
            "B-RetainedEarningsAccumulatedDeficit",
            "I-RetainedEarningsAccumulatedDeficit",
            "B-RevenueFromContractWithCustomerExcludingAssessedTax",
            "I-RevenueFromContractWithCustomerExcludingAssessedTax",
            "B-Revenues",
            "I-Revenues",
            "B-SellingAndMarketingExpense",
            "I-SellingAndMarketingExpense",
            "B-SellingGeneralAndAdministrativeExpense",
            "I-SellingGeneralAndAdministrativeExpense",
            "B-ShareBasedCompensation",
            "I-ShareBasedCompensation",
            "B-SharesOutstanding",
            "I-SharesOutstanding",
            "B-StockRepurchaseProgramAuthorizedAmount1",
            "I-StockRepurchaseProgramAuthorizedAmount1",
            "B-StockholdersEquity",
            "I-StockholdersEquity",
            "B-StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "I-StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "B-TreasuryStockValue",
            "I-TreasuryStockValue",
            "B-WeightedAverageNumberOfDilutedSharesOutstanding",
            "I-WeightedAverageNumberOfDilutedSharesOutstanding",
            "B-WeightedAverageNumberOfSharesOutstandingBasic",
            "I-WeightedAverageNumberOfSharesOutstandingBasic",
        ]

        tag2idx = {tag: int(i) for i, tag in enumerate(dataset_tags)}
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}
        return tag2idx, idx2tag

    # ==========================================================================
    # 核心接口：重写这个函数来适配你的 LLM
    # ==========================================================================
    
    def call_api(self, tokens: List[str]) -> List[str]:
        """
        【重要】这是需要你重写的核心函数！
        
        功能：调用你的 LLM API，对输入的 tokens 进行 NER 标注
        
        Args:
            tokens: 输入的 token 列表，例如:
                   ["The", "company", "reported", "revenue", "of", "$", "1.2", "million"]
        
        Returns:
            labels: 预测的标签列表，例如:
                   ["O", "O", "O", "B-Revenues", "O", "O", "I-Revenues", "I-Revenues"]
        
        注意：
        1. 返回的标签数量必须与输入 tokens 数量相同
        2. 标签格式必须是 IOB2 (B-EntityType, I-EntityType, O)
        3. 如果出错，返回全 "O" 标签
        
        示例实现见下方的 call_api_example_*() 函数
        """
        
        # 默认实现：返回全 O（请重写此函数！）
        LOGGER.warning("使用默认的 call_api 实现（全 O 标签）。请重写此函数以调用你的 LLM！")
        return ["O"] * len(tokens)
    
    # ==========================================================================
    # 示例实现：可以参考这些示例来重写 call_api
    # ==========================================================================
    
    def call_api_example_openai(self, tokens: List[str]) -> List[str]:
        """
        示例 1: 使用 OpenAI API
        
        复制此函数的内容到 call_api() 中，并修改参数
        """
        from openai import OpenAI
        
        # 初始化客户端（建议在 __init__ 中初始化）
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
        # 构造 prompt
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(tokens)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # 或 "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=2000
            )
            
            output = response.choices[0].message.content
            labels = self.parse_llm_output(output, len(tokens))
            return labels
            
        except Exception as e:
            LOGGER.error(f"OpenAI API 调用失败: {e}")
            return ["O"] * len(tokens)
    
    def call_api_example_custom(self, tokens: List[str]) -> List[str]:
        """
        示例 2: 使用自定义 API
        
        适用于任何兼容 OpenAI 格式的 API（如本地模型、其他云服务等）
        """
        import requests
        
        api_endpoint = os.getenv("LLM_API_ENDPOINT")
        api_key = os.getenv("LLM_API_KEY")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "your-model-name",
            "messages": [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": self.get_user_prompt(tokens)}
            ],
            "temperature": 0
        }
        
        try:
            response = requests.post(api_endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # 根据你的 API 响应格式调整
            output = result["choices"][0]["message"]["content"]
            labels = self.parse_llm_output(output, len(tokens))
            return labels
            
        except Exception as e:
            LOGGER.error(f"API 调用失败: {e}")
            return ["O"] * len(tokens)
    
    def call_api_example_claude(self, tokens: List[str]) -> List[str]:
        """
        示例 3: 使用 Anthropic Claude API
        """
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        try:
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0,
                system=self.get_system_prompt(),
                messages=[
                    {"role": "user", "content": self.get_user_prompt(tokens)}
                ]
            )
            
            output = message.content[0].text
            labels = self.parse_llm_output(output, len(tokens))
            return labels
            
        except Exception as e:
            LOGGER.error(f"Claude API 调用失败: {e}")
            return ["O"] * len(tokens)
    
    # ==========================================================================
    # Prompt 生成（可以根据需要自定义）
    # ==========================================================================
    
    def get_system_prompt(self, use_few_shot: bool = True) -> str:
        """
        生成系统 prompt
        
        你可以重写此函数来自定义 prompt
        """
        base_prompt = """You are an expert in financial document analysis and named entity recognition.

Your task is to identify and label financial entities in text using the IOB2 tagging scheme.

Entity Types (139 types including):
- Revenues, Assets, Liabilities, NetIncomeLoss, CashAndCashEquivalents
- StockholdersEquity, OperatingIncomeLoss, GrossProfit, etc.

Tagging Scheme:
- B-EntityType: Beginning of an entity
- I-EntityType: Inside/continuation of an entity  
- O: Outside any entity (not an entity)

Important Rules:
1. Return ONLY a JSON object: {"labels": ["label1", "label2", ...]}
2. The number of labels MUST match the number of input tokens
3. Financial numbers should be tagged based on their context
4. Multi-token entities: first token gets B-, subsequent tokens get I-
"""
        
        if use_few_shot:
            examples = """
Examples:

Input: ["Net", "income", "was", "$", "45.6", "million"]
Output: {"labels": ["B-NetIncomeLoss", "I-NetIncomeLoss", "O", "O", "I-NetIncomeLoss", "I-NetIncomeLoss"]}

Input: ["Total", "assets", "of", "$", "100", "million"]  
Output: {"labels": ["B-Assets", "I-Assets", "O", "O", "I-Assets", "I-Assets"]}

Input: ["The", "company", "reported", "revenue", "of", "$", "500", "million"]
Output: {"labels": ["O", "O", "O", "B-Revenues", "O", "O", "I-Revenues", "I-Revenues"]}

Input: ["Cash", "increased", "to", "$", "1.2", "billion"]
Output: {"labels": ["B-Cash", "O", "O", "O", "I-Cash", "I-Cash"]}
"""
            base_prompt += examples
        
        return base_prompt
    
    def get_user_prompt(self, tokens: List[str]) -> str:
        """生成用户 prompt"""
        return f"""Tag these {len(tokens)} tokens. Return exactly {len(tokens)} labels.

Input: {json.dumps(tokens)}
Output:"""
    
    # ==========================================================================
    # 输出解析
    # ==========================================================================
    
    def parse_llm_output(self, output: str, expected_length: int) -> List[str]:
        """
        解析 LLM 输出
        
        支持多种格式：
        1. JSON: {"labels": [...]}
        2. 列表: ["O", "B-Revenues", ...]
        3. 逐行: token label\ntoken label\n...
        """
        
        # 方法1: 提取 JSON
        try:
            json_match = re.search(r'\{[^}]*"labels"[^}]*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                labels = data.get("labels", [])
                if len(labels) == expected_length:
                    return self.validate_and_fix_labels(labels, expected_length)
        except Exception as e:
            LOGGER.debug(f"JSON 解析失败: {e}")
        
        # 方法2: 提取列表
        try:
            list_match = re.search(r'\[(.*?)\]', output, re.DOTALL)
            if list_match:
                content = list_match.group(1)
                labels = [s.strip().strip('"\'') for s in content.split(',')]
                if len(labels) == expected_length:
                    return self.validate_and_fix_labels(labels, expected_length)
        except Exception as e:
            LOGGER.debug(f"列表解析失败: {e}")
        
        # 方法3: 逐行解析
        try:
            lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
            labels = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    labels.append(parts[-1])
            if len(labels) == expected_length:
                return self.validate_and_fix_labels(labels, expected_length)
        except Exception as e:
            LOGGER.debug(f"逐行解析失败: {e}")
        
        # 如果都失败，返回全 O
        LOGGER.warning(f"无法解析 LLM 输出，返回全 O。输出前 200 字符: {output[:200]}")
        return ["O"] * expected_length
    
    def validate_and_fix_labels(self, labels: List[str], expected_length: int) -> List[str]:
        """验证并修复标签"""
        
        # 长度检查
        if len(labels) != expected_length:
            LOGGER.warning(f"标签数量不匹配: 期望 {expected_length}, 实际 {len(labels)}")
            if len(labels) < expected_length:
                labels.extend(["O"] * (expected_length - len(labels)))
            else:
                labels = labels[:expected_length]
        
        # 格式检查和修复
        fixed_labels = []
        for label in labels:
            label = label.strip()
            # 检查是否是有效标签
            if label in self.tag2idx:
                fixed_labels.append(label)
            elif label.upper() == "O":
                fixed_labels.append("O")
            else:
                # 尝试修复常见错误
                if label.startswith("B_") or label.startswith("I_"):
                    label = label.replace("_", "-", 1)
                    if label in self.tag2idx:
                        fixed_labels.append(label)
                    else:
                        fixed_labels.append("O")
                else:
                    LOGGER.debug(f"无效标签: {label}, 替换为 O")
                    fixed_labels.append("O")
        
        return fixed_labels
    
    # ==========================================================================
    # 评估流程
    # ==========================================================================
    
    def evaluate(
        self,
        split: str = "test",
        max_samples: int = None,
        save_predictions: bool = False
    ) -> Dict:
        """
        在数据集上运行评估
        
        Args:
            split: 数据集分割 ("train", "validation", "test")
            max_samples: 最大样本数（用于快速测试）
            save_predictions: 是否保存详细预测结果
        
        Returns:
            评估结果字典
        """
        LOGGER.info(f"加载 {split} 数据集...")
        
        # 加载数据集
        try:
            dataset = datasets.load_dataset(
                "nlpaueb/finer-139",
                split=split,
                revision="refs/convert/parquet"
            )
            LOGGER.info(f"成功从 HuggingFace Hub 加载数据集")
        except Exception as e1:
            LOGGER.warning(f"从 Hub 加载失败: {e1}")
            try:
                from huggingface_hub import hf_hub_download
                file_map = {
                    "train": "train-00000-of-00001.parquet",
                    "validation": "validation-00000-of-00001.parquet",
                    "test": "test-00000-of-00001.parquet",
                }
                local_file = hf_hub_download(
                    repo_id="nlpaueb/finer-139",
                    filename=file_map[split],
                    repo_type="dataset",
                )
                dataset = datasets.load_dataset("parquet", data_files=local_file, split="train")
                LOGGER.info(f"成功从 parquet 文件加载数据集")
            except Exception as e2:
                LOGGER.error(f"加载数据集失败: {e2}")
                raise
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        LOGGER.info(f"开始评估 {len(dataset)} 个样本...")
        
        y_true_all = []
        y_pred_all = []
        predictions = [] if save_predictions else None
        
        for idx, sample in enumerate(tqdm(dataset, desc="评估中")):
            tokens = sample["tokens"]
            true_labels_idx = sample["ner_tags"]
            
            # 转换为标签字符串
            true_labels = [self.idx2tag[label_idx] for label_idx in true_labels_idx]
            
            # 调用 LLM API 获取预测
            pred_labels = self.call_api(tokens)
            
            # 确保长度匹配
            if len(pred_labels) != len(true_labels):
                LOGGER.warning(
                    f"样本 {idx}: 预测长度 ({len(pred_labels)}) != 真实长度 ({len(true_labels)})"
                )
                if len(pred_labels) < len(true_labels):
                    pred_labels.extend(["O"] * (len(true_labels) - len(pred_labels)))
                else:
                    pred_labels = pred_labels[:len(true_labels)]
            
            y_true_all.append(true_labels)
            y_pred_all.append(pred_labels)
            
            if save_predictions:
                predictions.append({
                    "id": sample.get("id", idx),
                    "tokens": tokens,
                    "true_labels": true_labels,
                    "pred_labels": pred_labels
                })
        
        # 计算评估指标
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info(f"{split.upper()} 数据集评估结果")
        LOGGER.info("=" * 80 + "\n")
        
        report = classification_report(
            y_true=y_true_all,
            y_pred=y_pred_all,
            zero_division=0,
            mode=None,
            digits=4,
            scheme=IOB2,
        )
        
        LOGGER.info(report)
        
        results = {
            "split": split,
            "num_samples": len(dataset),
            "report": report
        }
        
        if save_predictions:
            results["predictions"] = predictions
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="使用 LLM 评估 FiNER-139 benchmark"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="数据集分割"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数（用于快速测试）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_evaluation_results.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="保存详细的预测结果"
    )
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = FiNERLLMEvaluator()
    
    # 提示用户重写 call_api
    LOGGER.info("=" * 80)
    LOGGER.info("注意：请确保已经重写了 call_api() 方法来调用你的 LLM！")
    LOGGER.info("参考 call_api_example_*() 方法了解如何实现")
    LOGGER.info("=" * 80 + "\n")
    
    # 运行评估
    results = evaluator.evaluate(
        split=args.split,
        max_samples=args.max_samples,
        save_predictions=args.save_predictions
    )
    
    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    LOGGER.info(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

