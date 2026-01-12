---
title: "临床数据分析与线性回归建模（PICU患者数据应用）"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/picu-linear-regression
date: 2026-01-11
excerpt: "基于PICU患者临床数据，完成数据清洗、特征可视化与线性回归建模，探索患者关键指标与临床结局的量化关系"
header:
  teaser: /images/portfolio/picu-linear-regression/numeric_features_hist.png
tags:
- 临床数据分析
- 线性回归
- 数据可视化
- 特征工程
tech_stack:
- name: Python
- name: Pandas
- name: Matplotlib
- name: Seaborn
- name: Statsmodels
---

## 项目背景
儿科重症监护室（PICU）患者的临床数据包含年龄、性别、实验室指标等多维度信息，对这些数据的深入分析可揭示关键指标与患者结局的关联。本项目基于13258条PICU患者数据，通过数据清洗、特征探索与线性回归建模，掌握临床数据处理核心流程，量化分析实验室指标与患者临床结局的线性关系，为临床风险评估提供数据支撑。

## 代码实现
### 编程库
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
```
### 1. 数据读取与预处理
聚焦关键临床指标，处理缺失值并筛选有效特征，为建模奠定基础。
```python
path = "icu_first24hours.csv"
picu_data = pd.read_csv(path)
picu_data
# 筛选感兴趣的特征集（年龄、结局变量、关键实验室指标）
core_features = [
    'age_month', 'HOSPITAL_EXPIRE_FLAG',
    'lab_5097_max', 'lab_5099_min', 'lab_5237_min',
    'lab_5227_min', 'lab_5225_range', 'lab_5235_max', 'lab_5257_min'
]
new_picu_data = picu_data[core_features].copy()
```
### 2. 统计分析
数据集探索，将感兴趣的指标进行统计和可视化
```python
# 查看各列的非空值数量 (count)
new_picu_data.count()
# 计算均值 (mean)
new_picu_data.mean()
# 计算中位数 (median)
new_picu_data.median()
# 计算方差 (var)
new_picu_data.var()
#new_picu_data.to_excel("data.xlsx", index=False)
```
| 变量 | 非空值数 (Count) | 均值 (Mean) | 中位数 (Median) | 方差 (Variance) |
|---|---:|---:|---:|---:|
| age_month | 13258 | 29.989 | 8.000 | 1939.011 |
| HOSPITAL_EXPIRE_FLAG | 13258 | 0.0588 | 0.000 | 0.0554 |
| lab_5097_max | 11346 | 38.270 | 36.900 | 80.800 |
| lab_5099_min | 11343 | 120.593 | 118.000 | 977.031 |
| lab_5237_min | 8668 | 7.347 | 7.368 | 0.013 |
| lab_5227_min | 8616 | 2.248 | 1.800 | 3.327 |
| lab_5225_range | 8666 | 3.842 | 1.500 | 29.846 |
| lab_5235_max | 8671 | 42.610 | 39.600 | 218.750 |
| lab_5257_min | 8662 | 91.546 | 105.000 | 3052.255 |

临床特征的直方图和相线图展示
```python
# 定义要绘制直方图的列名列表，包含8个临床指标
colname =['age_month','lab_5097_max','lab_5099_min','lab_5237_min','lab_5227_min','lab_5225_range','lab_5235_max','lab_5257_min']
fig, axs = plt.subplots(int(len(colname)/2), 2, constrained_layout=True, figsize=(8, 6), dpi=150)
for i in range(len(colname)):
    # 核心代码：使用seaborn的histplot函数绘制直方图
    # x=colname[i] 指定要绘制的变量（当前循环的列名）
    # data=picu_data 指定数据源
    # alpha=0.4 设置直方图的透明度为0.4（0为完全透明，1为完全不透明）
    # kde=True 在直方图上叠加核密度估计曲线，显示数据的概率密度分布
    # ax=axs[i//2, i%2] 指定当前直方图绘制在哪个子图位置
    # i//2 计算行索引（整数除法），i%2 计算列索引（取余数）
    sns.histplot(x=colname[i], data=new_picu_data, alpha=0.4, kde=True, ax=axs[i//2, i%2])
plt.suptitle("Bar Plot")
```

### 2. 统计分析


