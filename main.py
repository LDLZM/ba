import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
import dgl
from scipy.io import loadmat
import yaml

logger = logging.getLogger(__name__)
# sys.path.append("..")


def get_gtan_args():
    yaml_file = "config/gtan_cfg.yaml"
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = 'gtan'
    return args
def main():
    args = get_gtan_args()
    from methods.gtan.gtan_main import gtan_main, load_gtan_data
    feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(
        args['dataset'], args['test_size'])
    # feat_data用于存储数据集的特征数据，通常是一个矩阵或数组，每一行代表一个样本的特征向量。
    # labels 标签
    # train_idx, 
    # test_idx,
    # g, 
    # cat_features 类别特征
    gtan_main(
        feat_data, g, train_idx, test_idx, labels, args, cat_features)

if __name__ == "__main__":
    main()