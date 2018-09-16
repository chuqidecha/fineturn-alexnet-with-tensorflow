#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-14 下午10:35
# @Author  : yinwb
# @File    : setting.py

TRAIN_LAYERS = ("fc6", "fc7", "fc8")

NUM_CLASS = 21

# 图像大小与通道数
IMAGE_SIZE = 227
IMAGE_CHANNEL = 3

# 神经网络配置参数
BATCH_SIZE = 128  # 小批量梯度下降每个batch的样本数
EPOCH = 100  # 训练轮数

LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNIING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 损失函数中模型参数正则化项的权重
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

TRAIN_NUM_EXAMPLES = 1680