"""
评估自定义 API 在 FiNER-139 数据集上的性能

使用方法:
    python evaluate_custom_api.py --api_endpoint "http://your-api-url" --split "test"

或者使用本地函数:
    python evaluate_custom_api.py --use_local_function --split "test"
"""

import argparse
import datasets
import itertools
import json
import logging
import requests
from typing import List, Dict, Tuple
from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


class CustomAPIEvaluator:
    """用于评估自定义 API 在 FiNER-139 benchmark 上的表现"""

    def __init__(self, api_endpoint: str = None, api_key: str = None):
        """
        初始化评估器

        Args:
            api_endpoint: API 端点 URL
            api_key: API 密钥（如果需要）
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.tag2idx, self.idx2tag = self.load_dataset_tags()

    @staticmethod
    def load_dataset_tags() -> Tuple[Dict, Dict]:
        """加载数据集标签 - 使用硬编码的标签列表（FiNER-139 的固定标签）"""
        # 由于新版本 datasets 库不支持脚本数据集，直接使用硬编码的标签列表
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

    def call_api(self, tokens: List[str]) -> List[str]:
        """
        调用 API 获取预测结果

        Args:
            tokens: 输入的 token 列表

        Returns:
            预测的标签列表

        API 输入格式示例:
        {
            "tokens": ["The", "company", "reported", "revenue", "of", "$", "1.2", "million"]
        }

        API 输出格式示例:
        {
            "labels": ["O", "O", "O", "B-Revenues", "O", "O", "I-Revenues", "I-Revenues"]
        }
        """
        if self.api_endpoint is None:
            raise ValueError("API endpoint not provided")

        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"tokens": tokens}

        try:
            response = requests.post(
                self.api_endpoint, json=payload, headers=headers, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # 根据你的 API 实际返回格式调整这里
            if "labels" in result:
                return result["labels"]
            elif "predictions" in result:
                return result["predictions"]
            else:
                raise ValueError(f"Unexpected API response format: {result}")

        except requests.exceptions.RequestException as e:
            LOGGER.error(f"API request failed: {e}")
            # 返回全 O 标签作为后备
            return ["O"] * len(tokens)

    def predict_local(self, tokens: List[str]) -> List[str]:
        """
        使用本地函数进行预测（用于测试或本地模型）

        你需要根据自己的模型修改这个函数

        Args:
            tokens: 输入的 token 列表

        Returns:
            预测的标签列表
        """
        # 示例：这里是一个简单的规则基线
        # 你应该替换成你自己的模型预测逻辑

        # 方法1: 如果你有本地的模型文件
        # from transformers import AutoModelForTokenClassification, AutoTokenizer
        # model = AutoModelForTokenClassification.from_pretrained("your_model_path")
        # tokenizer = AutoTokenizer.from_pretrained("your_model_path")
        # ...进行预测...

        # 方法2: 如果你想加载训练好的模型
        # import tensorflow as tf
        # model = tf.keras.models.load_model("your_model_path")
        # ...进行预测...

        # 临时示例代码（返回全 O）
        LOGGER.warning(
            "Using default prediction (all 'O' labels). Please implement your own prediction logic."
        )
        return ["O"] * len(tokens)

    def evaluate(
        self,
        split: str = "test",
        max_samples: int = None,
        use_local: bool = False,
        batch_size: int = 1,
    ) -> Dict:
        """
        在指定的数据集分割上评估 API 性能

        Args:
            split: 数据集分割 ("train", "validation", "test")
            max_samples: 最大样本数（用于快速测试）
            use_local: 是否使用本地预测函数
            batch_size: 批处理大小（如果你的 API 支持批处理）

        Returns:
            评估结果字典
        """
        LOGGER.info(f"Loading {split} dataset from FiNER-139...")

        # 尝试多种方式加载数据集
        dataset = None

        # 方法1: 尝试直接从 HuggingFace Hub 加载（使用 revision 参数）
        try:
            LOGGER.info(f"Attempting to load from HuggingFace Hub...")
            dataset = datasets.load_dataset(
                "nlpaueb/finer-139",
                split=split,
                revision="refs/convert/parquet",  # 使用 parquet 转换分支
            )
            LOGGER.info(
                f"Successfully loaded {len(dataset)} samples from HuggingFace Hub"
            )
        except Exception as e1:
            LOGGER.warning(f"Method 1 failed: {e1}")

            # 方法2: 尝试使用 data_files 直接加载 parquet
            try:
                LOGGER.info(f"Attempting to load from parquet files...")
                # 检查 HuggingFace 上实际的文件路径
                parquet_urls = {
                    "train": "hf://datasets/nlpaueb/finer-139/train-00000-of-00001.parquet",
                    "validation": "hf://datasets/nlpaueb/finer-139/validation-00000-of-00001.parquet",
                    "test": "hf://datasets/nlpaueb/finer-139/test-00000-of-00001.parquet",
                }

                if split not in parquet_urls:
                    raise ValueError(
                        f"Invalid split: {split}. Must be one of: train, validation, test"
                    )

                dataset = datasets.load_dataset(
                    "parquet", data_files=parquet_urls[split], split="train"
                )
                LOGGER.info(f"Successfully loaded {len(dataset)} samples from parquet")
            except Exception as e2:
                LOGGER.warning(f"Method 2 failed: {e2}")

                # 方法3: 下载并缓存 parquet 文件
                try:
                    LOGGER.info(f"Attempting to download parquet file...")
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

                    dataset = datasets.load_dataset(
                        "parquet", data_files=local_file, split="train"
                    )
                    LOGGER.info(
                        f"Successfully loaded {len(dataset)} samples from downloaded file"
                    )
                except Exception as e3:
                    LOGGER.error(f"All methods failed to load dataset.")
                    LOGGER.error(f"Method 1 error: {e1}")
                    LOGGER.error(f"Method 2 error: {e2}")
                    LOGGER.error(f"Method 3 error: {e3}")
                    LOGGER.info(
                        "\nPlease try one of the following:\n"
                        "1. Check your internet connection\n"
                        "2. Manually download the dataset from https://huggingface.co/datasets/nlpaueb/finer-139\n"
                        "3. Use the original finer.py with an older version of datasets library\n"
                    )
                    raise RuntimeError(
                        "Failed to load dataset after trying all methods"
                    )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        LOGGER.info(f"Evaluating on {len(dataset)} samples...")

        y_true_all = []
        y_pred_all = []

        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            tokens = sample["tokens"]
            true_labels_idx = sample["ner_tags"]

            # 将索引转换为标签字符串
            true_labels = [self.idx2tag[label_idx] for label_idx in true_labels_idx]

            # 调用 API 或本地函数获取预测
            if use_local:
                pred_labels = self.predict_local(tokens)
            else:
                pred_labels = self.call_api(tokens)

            # 确保预测标签数量与真实标签数量一致
            if len(pred_labels) != len(true_labels):
                LOGGER.warning(
                    f"Sample {idx}: Prediction length ({len(pred_labels)}) "
                    f"doesn't match true labels ({len(true_labels)}). Padding/truncating."
                )
                if len(pred_labels) < len(true_labels):
                    pred_labels.extend(["O"] * (len(true_labels) - len(pred_labels)))
                else:
                    pred_labels = pred_labels[: len(true_labels)]

            y_true_all.append(true_labels)
            y_pred_all.append(pred_labels)

        # 计算评估指标
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info(f"Evaluation Results on {split.upper()} Split")
        LOGGER.info("=" * 80 + "\n")

        # 使用 seqeval 计算详细的分类报告
        report = classification_report(
            y_true=y_true_all,
            y_pred=y_pred_all,
            zero_division=0,
            mode=None,
            digits=4,
            scheme=IOB2,
        )

        LOGGER.info(report)

        # 保存结果
        results = {"split": split, "num_samples": len(dataset), "report": report}

        return results

    def evaluate_sample(
        self, tokens: List[str], true_labels: List[str] = None, use_local: bool = False
    ) -> Dict:
        """
        评估单个样本（用于调试）

        Args:
            tokens: token 列表
            true_labels: 真实标签列表（如果有）
            use_local: 是否使用本地预测

        Returns:
            包含预测结果的字典
        """
        if use_local:
            pred_labels = self.predict_local(tokens)
        else:
            pred_labels = self.call_api(tokens)

        result = {"tokens": tokens, "predictions": pred_labels}

        if true_labels:
            result["true_labels"] = true_labels
            # 计算单个样本的准确率
            correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
            result["token_accuracy"] = correct / len(tokens)

        return result


def main():
    parser = argparse.ArgumentParser(
        description="评估自定义 API 在 FiNER-139 benchmark 上的性能"
    )
    parser.add_argument(
        "--api_endpoint",
        type=str,
        default=None,
        help="API 端点 URL (例如: http://localhost:8000/predict)",
    )
    parser.add_argument(
        "--api_key", type=str, default=None, help="API 密钥（如果需要认证）"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="要评估的数据集分割",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="最大评估样本数（用于快速测试）"
    )
    parser.add_argument(
        "--use_local_function", action="store_true", help="使用本地预测函数而不是 API"
    )
    parser.add_argument(
        "--demo", action="store_true", help="运行演示模式（评估单个样本）"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json", help="输出结果文件路径"
    )

    args = parser.parse_args()

    # 初始化评估器
    evaluator = CustomAPIEvaluator(api_endpoint=args.api_endpoint, api_key=args.api_key)

    if args.demo:
        # 演示模式：评估单个示例
        LOGGER.info("Running in demo mode...")
        demo_tokens = [
            "The",
            "company",
            "reported",
            "revenue",
            "of",
            "$",
            "1.2",
            "million",
        ]
        demo_labels = [
            "O",
            "O",
            "O",
            "B-Revenues",
            "O",
            "O",
            "I-Revenues",
            "I-Revenues",
        ]

        result = evaluator.evaluate_sample(
            tokens=demo_tokens,
            true_labels=demo_labels,
            use_local=args.use_local_function,
        )

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Demo Sample Evaluation")
        LOGGER.info("=" * 80)
        LOGGER.info(f"\nTokens: {result['tokens']}")
        LOGGER.info(f"True Labels: {result['true_labels']}")
        LOGGER.info(f"Predictions: {result['predictions']}")
        LOGGER.info(f"Token Accuracy: {result['token_accuracy']:.4f}")

    else:
        # 完整评估模式
        results = evaluator.evaluate(
            split=args.split,
            max_samples=args.max_samples,
            use_local=args.use_local_function,
        )

        # 保存结果
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
