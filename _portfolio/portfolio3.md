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

## 项目背景 (Background)
儿科重症监护室（PICU）患者的临床数据包含年龄、性别、实验室指标等多维度信息，对这些数据的深入分析可揭示关键指标与患者结局的关联。本项目基于13258条PICU患者数据，通过数据清洗、特征探索与线性回归建模，掌握临床数据处理核心流程，量化分析实验室指标与患者临床结局的线性关系，为临床风险评估提供数据支撑。

## 核心实现 (Implementation)
### 1. 数据预处理与特征筛选（核心逻辑）
聚焦关键临床指标，处理缺失值并筛选有效特征，为建模奠定基础。
```python
# 筛选核心特征集（年龄、结局变量、关键实验室指标）
core_features = [
    'age_month', 'HOSPITAL_EXPIRE_FLAG',
    'lab_5097_max', 'lab_5099_min', 'lab_5237_min',
    'lab_5227_min', 'lab_5225_range', 'lab_5235_max', 'lab_5257_min'
]
new_picu_data = picu_data[core_features].copy()

# 缺失值处理（核心代码）
# 数值特征用中位数填充
numeric_cols = new_picu_data.columns[new_picu_data.dtypes != 'object']
new_picu_data[numeric_cols] = new_picu_data[numeric_cols].fillna(new_picu_data[numeric_cols].median())

# 验证数据完整性
print("处理后各列非空值数量：")
print(new_picu_data.count())
