---
title: "机器学习分类与聚类实战（临床数据应用）"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/classification-clustering-clinical-data
date: 2026-01-12
excerpt: "基于13258条临床数据，实现逻辑回归、随机森林、SVM三大分类模型与K-Means聚类，解决患者死亡风险预测与临床指标分组问题"
header:
  teaser: /images/portfolio/classification-clustering/elbow_method.png
tags:
- 分类算法
- 无监督聚类
- 临床数据挖掘
- 不平衡数据处理
tech_stack:
- name: Python
- name: Scikit-learn
- name: Pandas
- name: Matplotlib
- name: Seaborn
---

## 项目背景 (Background)
临床数据包含患者年龄、实验室指标等多维度信息，通过机器学习可实现两大核心目标：一是预测患者住院死亡风险（分类任务），辅助临床风险评估；二是基于实验室指标对患者进行无监督分组（聚类任务），挖掘潜在临床亚型。本项目基于13258条患者数据，系统实现数据预处理、多模型分类、K-Means聚类及效果评估，完整覆盖机器学习核心流程。

## 核心实现 (Implementation)
### 1. 数据预处理（核心逻辑）
完成缺失值填充、数据集划分与标准化，为建模提供高质量数据。
