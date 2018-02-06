# -*- coding: utf-8 -*-
from datetime import datetime

import pandas as pd
import tensorflow as tf
from tensorflow.python.estimator.inputs.pandas_io import pandas_input_fn

from utils.multiapplier import mg, apply_by_multiprocessing
from utils.normalizer import norm_by_mean

raw_data = pd.read_csv('csv/clean_data_sample.csv', sep='\t', index_col='HouseID', encoding='utf-8')
fmt_data = raw_data.drop(columns=['RealEstateProjectID', 'ParkingSpaceAmount.1', 'Unnamed: 0']).fillna(0)

# Info
ifo_data = fmt_data[['ProjectName', 'ClassifyResult', 'HouseUseType']]

# 热编码处理
tsf_data = fmt_data.drop(columns=['ProjectName', 'ClassifyResult', 'HouseUseType'])
tsf_data = pd.get_dummies(tsf_data)

# 归一化处理
cpu_count = mg.cpu_count() - 1 if mg.cpu_count() > 1 else mg.cpu_count
tsf_data = apply_by_multiprocessing(tsf_data, norm_by_mean, axis=1, workers=cpu_count)

# 归一化数据留存
dupe = pd.concat([ifo_data, tsf_data], axis=1)
dupe.to_csv('csv/clean_data_sample_normalize_{}'.format(datetime.now()))

# 训练样本
train_x = tsf_data
train_y = ifo_data['ClassifyResult']


def main():
    xuzhou_feature_columns = []

    classifier = tf.estimator.DNNClassifier(
        feature_columns=xuzhou_feature_columns,
        hidden_units=[40, 20],
        n_classes=3)

    classifier.train(
        input_fn=lambda: pandas_input_fn(train_x, train_y, ),
        steps=1)
