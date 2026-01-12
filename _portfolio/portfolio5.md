---
title: "MNIST手写数字识别：CNN实战与性能优化"
collection: portfolio
type: "Deep Learning"
permalink: /portfolio/mnist-cnn-handwritten-digit-recognition
date: 2026-01-12
excerpt: "基于TensorFlow/Keras构建卷积神经网络（CNN），解决MNIST手写数字识别问题，通过架构优化与正则化技术实现99.32%测试准确率"
tags:
- 卷积神经网络
- 图像分类
- 手写数字识别
- 深度学习
- 模型优化
tech_stack:
- name: Python
- name: TensorFlow/Keras
- name: NumPy
- name: Matplotlib
- name: Seaborn
---

## 项目背景
MNIST（Modified National Institute of Standards and Technology）数据集是计算机视觉领域的经典入门数据集，广泛用于图像分类算法的基准测试。该数据集包含70,000张28×28像素的灰度手写数字图像，其中训练集60,000张、测试集10,000张，涵盖0-9共10个数字类别。

### 核心挑战与目标
- 手写数字存在书写风格差异、倾斜变形、笔画粗细不均等问题，需模型具备强鲁棒性；
- 基础机器学习模型（如朴素贝叶斯）在MNIST上准确率仅约81.49%，需通过CNN的层级特征提取能力提升性能；
- 项目目标：构建高效CNN模型，实现99%以上测试准确率，同时优化训练效率与泛化能力。

## 代码实现
