# -*- coding: utf-8 -*-
from datetime import datetime

import argparse

import pandas as pd
import tensorflow as tf
from tensorflow.python.estimator.inputs.pandas_io import pandas_input_fn

from utils.multiapplier import mg, apply_by_multiprocessing
from utils.normalizer import norm_by_mean


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--train_steps', default=30, type=int,
                    help='number of training steps')


def load_data():
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
    return train_x, train_y


def main(argv):
    args = parser.parse_args(argv[1:])

    train_x, train_y = load_data()
    xuzhou_feature_columns = []

    for i, colname in enumerate(train_x.columns):
        xuzhou_feature_columns.append(colname)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=xuzhou_feature_columns,
        hidden_units=[40, 20],
        model_dir='/tmp',
        n_classes=3)

    classifier.train(
        input_fn=lambda: pandas_input_fn(train_x, y=train_y, batch_size=args.batch_size),
        steps=args.train_steps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
