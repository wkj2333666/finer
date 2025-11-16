#!/usr/bin/env python3
"""
测试 API 服务器的简单脚本

使用方法:
    python test_api.py --endpoint "http://localhost:8000/predict"
"""

import argparse
import requests
import json


def test_single_request(endpoint: str, tokens: list):
    """测试单个请求"""
    print(f"\n{'='*60}")
    print("Testing single prediction request")
    print(f"{'='*60}")
    print(f"Endpoint: {endpoint}")
    print(f"Input tokens: {tokens}")
    
    try:
        response = requests.post(
            endpoint,
            json={"tokens": tokens},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        response.raise_for_status()
        result = response.json()
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response:")
        print(json.dumps(result, indent=2))
        
        if "labels" in result:
            labels = result["labels"]
            print(f"\n{'Token':<20} {'Label':<20}")
            print("-" * 40)
            for token, label in zip(tokens, labels):
                print(f"{token:<20} {label:<20}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")
        return False


def test_health_check(base_url: str):
    """测试健康检查端点"""
    print(f"\n{'='*60}")
    print("Testing health check endpoint")
    print(f"{'='*60}")
    
    health_url = base_url.replace("/predict", "/health")
    
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False


def test_multiple_samples(endpoint: str):
    """测试多个样本"""
    print(f"\n{'='*60}")
    print("Testing multiple samples")
    print(f"{'='*60}")
    
    test_samples = [
        {
            "tokens": ["The", "company", "reported", "revenue", "of", "$", "1.2", "million"],
            "description": "Revenue example"
        },
        {
            "tokens": ["Total", "assets", "were", "$", "500", "million"],
            "description": "Assets example"
        },
        {
            "tokens": ["Net", "income", "increased", "by", "15", "%"],
            "description": "Income example"
        },
        {
            "tokens": ["The", "company", "has", "debt", "of", "$", "200", "million"],
            "description": "Debt/Liabilities example"
        }
    ]
    
    success_count = 0
    for i, sample in enumerate(test_samples, 1):
        print(f"\n[Sample {i}] {sample['description']}")
        print(f"Tokens: {sample['tokens']}")
        
        try:
            response = requests.post(
                endpoint,
                json={"tokens": sample["tokens"]},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if "labels" in result:
                labels = result["labels"]
                print(f"Labels: {labels}")
                
                # 检查是否识别到了实体
                non_o_labels = [l for l in labels if l != 'O']
                if non_o_labels:
                    print(f"✓ Entities detected: {non_o_labels}")
                else:
                    print("✗ No entities detected")
                
                success_count += 1
            else:
                print(f"✗ Unexpected response format: {result}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(test_samples)} tests passed")
    print(f"{'='*60}")
    
    return success_count == len(test_samples)


def test_error_handling(endpoint: str):
    """测试错误处理"""
    print(f"\n{'='*60}")
    print("Testing error handling")
    print(f"{'='*60}")
    
    test_cases = [
        {
            "payload": {},
            "description": "Empty payload"
        },
        {
            "payload": {"tokens": []},
            "description": "Empty tokens list"
        },
        {
            "payload": {"wrong_key": ["test"]},
            "description": "Wrong key name"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['description']}")
        print(f"Payload: {test['payload']}")
        
        try:
            response = requests.post(
                endpoint,
                json=test["payload"],
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code >= 400:
                print("✓ Error handled correctly")
            else:
                print("✗ Should return error status")
                
        except Exception as e:
            print(f"Exception: {e}")


def main():
    parser = argparse.ArgumentParser(description="测试 NER API 服务器")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/predict",
        help="API 端点 URL"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["single", "health", "multiple", "error", "all"],
        help="选择要运行的测试"
    )
    
    args = parser.parse_args()
    
    base_url = args.endpoint.split("/predict")[0] if "/predict" in args.endpoint else args.endpoint
    
    print(f"\n{'#'*60}")
    print(f"# NER API Testing Script")
    print(f"# API Endpoint: {args.endpoint}")
    print(f"{'#'*60}")
    
    results = {}
    
    if args.test in ["health", "all"]:
        results["health_check"] = test_health_check(base_url)
    
    if args.test in ["single", "all"]:
        test_tokens = ["The", "company", "reported", "revenue", "of", "$", "1.2", "million"]
        results["single_request"] = test_single_request(args.endpoint, test_tokens)
    
    if args.test in ["multiple", "all"]:
        results["multiple_samples"] = test_multiple_samples(args.endpoint)
    
    if args.test in ["error", "all"]:
        results["error_handling"] = test_error_handling(args.endpoint)
    
    # 总结
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

