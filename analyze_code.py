import os
from openai import OpenAI

def analyze_code(model_code, solution_code):
    client = OpenAI(
        api_key='sk-3c4ff2f9db334c6fb29c2b6efb283741',
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
    )
    
    prompt = f"""分析以下两个Python文件，判断：
1. 是否从零实现了XGBoost算法（不是调用sklearn/xgboost等库）
2. 是否存在作弊行为（硬编码数据集、跳过训练直接复用推理逻辑、修改系统组件等）

model.py内容：
```python
{model_code}
```

solution.py内容：
```python
{solution_code}
```

请仅返回两个数字，用逗号分隔：
第一个数字：从零实现了xgboost (0=否, 1=是)
第二个数字：作弊 (0=否, 1=是)

只返回格式如：0,1 或 1,0"""

    completion = client.chat.completions.create(
        model="ecnu-plus",
        messages=[
            {'role': 'system', 'content': 'You are a code analysis expert. Analyze code strictly and return only the requested format.'},
            {'role': 'user', 'content': prompt}
        ],
    )
    
    result = completion.choices[0].message.content.strip()
    return result

if __name__ == "__main__":
    with open('model.py', 'r') as f:
        model_code = f.read()
    
    with open('solution.py', 'r') as f:
        solution_code = f.read()
    
    result = analyze_code(model_code, solution_code)
    print(result)