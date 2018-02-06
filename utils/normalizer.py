# -*- coding: utf-8 -*-
# 云房数据标准化工具

__author__ = 'Gsy.HonglinHuang'


def norm_by_mean(x, mean, max_val, min_val):
    """
    近似归一化
    :return: 结果序列
    """
    x = float(x)
    norm = round((x - mean) / (max_val - min_val), 2)
    return norm


