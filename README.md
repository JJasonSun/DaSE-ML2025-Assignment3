
# Assignment 3: 网络入侵检测分类任务 (XGBoost)

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/License-Educational-green.svg)](https://www.guangmingai.com/chat/LICENSE)

**📅 截止日期：11月30日** | **🏆 [查看实时排行榜](http://101.132.193.95:3000/)**

---

## 📋 任务概述

> **核心目标**：基于网络流量特征，从零实现并优化 **XGBoost** 算法，完成入侵检测二分类任务（正常流量 vs 攻击流量）。

### 🎯 核心挑战
1.  **算法实现**：从零实现梯度提升 (Gradient Boosting) 算法逻辑。
2.  **精度达标**：模型 ROC-AUC 必须达到 **0.98** 以上。
3.  **工程优化**：在保证高精度的前提下，通过并行化、算法剪枝等手段极致优化 **训练与推理速度**。

### 📤 强制提交要求:
1. 必须通过「水杉平台」提交，未在该平台提交或提交超时视为无效
2. 水杉平台每人只能使用12小时
---

## 📂 文件结构与任务清单

### 1. 我提供的基础框架
| 文件名 | 说明 |
| :--- | :--- |
| **`model.py`** | 当前仅包含一个简单的逻辑回归模型 (Baseline)。**你需要在此处重写代码。** |
| **`solution.py`** | 包含 `fit` (训练) 和 `forward` (推理) 接口，负责数据预处理和模型调用。 |
| **`evaluate_local.py`** | 非提交程序，仅用于展示评测框架内部逻辑。 |
| **`evaluate_linux`** | 计算 ROC-AUC 和 Latency，并将结果提交至服务器。 |
| **`*.csv`** | `train.csv` (训练数据)，`test_local.csv`(本地测试) _和 `encrypt_test.csv` (加密测试数据)。 |

### 2. 你需要完成的工作
* **✨ 实现 XGBoost 模型**
    * 修改 `model.py`，用 **XGBoost (eXtreme Gradient Boosting)** 算法替换现有的逻辑回归。
* **✨ 适配 Solution 接口**
    * 修改 `solution.py`，确保 `fit` 和 `forward` 能正确驱动你的 XGBoost 模型。
* **✨ 性能调优 (关键)**
    * 确保 **ROC-AUC ≥ 0.98**。
    * 应用并行计算、直方图优化等技巧，缩短训练和推理时间以获取高分。
* **✨ 提交与冲榜**
    * 运行 `evaluate-linux` 进行测试和自动提交。

---

## 📈 评分标准 (总分 20分)

本次作业总分为 20 分。评分核心逻辑是：**在满足精度门槛 (AUC ≥ 0.98) 的前提下，根据运行效率 (Latency) 动态定分**。

### 1️⃣ Latency 计算方式
为了综合考量算法的训练效率与线上推理能力，**Latency** 定义为训练时间和测试时间的**几何平均数**（越低越好）：

$$\text{Latency} = \sqrt{\text{Training Time} \times \text{Testing Time}}$$

* **Training Time**: `fit()` 函数的完整运行时间。
* **Testing Time**: 完成所有测试样本 `forward()` 推理的总时间。

### 2️⃣ 详细得分规则
我们将统计所有 **达标 (AUC≥0.98)** 提交的 Latency，计算出 **前10%分位数 (P10)** 作为满分基准线。

| 场景 | 条件说明 | 最终得分 |
| :--- | :--- | :--- |
| **1** | ROC-AUC < 0.98 | **6 分** |
| **2** | AUC ≥ 0.98 且 Latency ≤ P10 | **20 分** |
| **3** | AUC ≥ 0.98 且 P10 < Latency ≤ 4×P10 | **6 ~ 20 分** |
| **4** | AUC ≥ 0.98 且 Latency > 4×P10 | **6 分** |

#### 🧮 线性插值公式
当你的速度处于 *2* 区间 ($P10 < Latency \le 2 \times P10$) 时，得分计算如下：

$$
\text{Total Score} = 6 + 14 \times \left( 1 - \frac{\text{Latency} - P10}{P10} \right)
$$

> **💡 策略提示**：
> * **精度第一**：若 AUC 不达标，无论多快都只有 6 分。
> * **速度决胜**：一旦 AUC 达标，分数完全取决于 Latency。若速度慢于基准线的 2 倍，即便精度很高也只能拿 6 分。

---

## ⚙️ 环境与运行指南

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

### 1. 安装依赖
推荐使用 Conda 管理环境：

```bash
conda create -n ML python=3.8
conda activate ML
````

### 2\. 设置环境变量 (必须)

在运行代码前，请根据你的系统设置个人身份信息。

**🐧 Linux/macOS:**

```bash
export STUDENT_ID='你的学号'
export STUDENT_NAME='你的姓名'
export STUDENT_NICKNAME='你的昵称'
export MAIN_CONTRIBUTOR='AI' 
```

**🪟 Windows PowerShell:**

```powershell
$env:STUDENT_ID="你的学号"
$env:STUDENT_NAME="你的姓名"
$env:STUDENT_NICKNAME="你的昵称"
$env:MAIN_CONTRIBUTOR="human"
```

### 3\. 运行评测

**本次作业仅在水杉平台提交**

```bash
chmod +x evaluate-linux
./evaluate-linux
```

**✅ 成功运行示例:**

```text
Training...
Training completed in 5.23s
Testing...
==================================================
Training Time: 5.23s
Testing Time:  0.45s
Latency:   1.53s  <-- 计算方式：sqrt(5.23 * 0.45)
ROC-AUC:       0.9850 <-- 必须 >= 0.98
==================================================

Submitting to leaderboard...
Submission successful!
```

-----

### 🎉 Good Luck\!

**请在截止日期前提交你的最佳优化版本！**

Made with ❤️ for Machine Learning Education

