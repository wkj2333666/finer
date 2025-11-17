baseline_prompt = """
Your task is to identify and label financial entities in text using the IOB2 tagging scheme.

Entity Types (139 types including):
- Revenues, Assets, Liabilities, NetIncomeLoss, CashAndCashEquivalents
- StockholdersEquity, OperatingIncomeLoss, GrossProfit, etc.

Tagging Scheme:
- B-EntityType: Beginning of an entity
- I-EntityType: Inside/continuation of an entity  
- O: Outside any entity (not an entity)

Important Rules:
1. Return ONLY a JSON object: ["label1", "label2", ...]
2. The number of labels MUST match the number of input tokens
3. Financial numbers should be tagged based on their context
4. Multi-token entities: first token gets B-, subsequent tokens get I-
"""
