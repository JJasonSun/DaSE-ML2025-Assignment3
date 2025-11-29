import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 自动加载当前目录的 .env 文件

def analyze_code(model_code, solution_code):
    # 从环境变量中读取 API key 和 base_url，避免在代码中明文写死
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://chat.ecnu.edu.cn/open/api/v1")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置，请在环境变量或 .env 文件中配置后再运行。")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
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