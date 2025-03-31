import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
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


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", default=str)  # specify which method to use
    method = vars(parser.parse_args())['method']  # dict

    # if method in ['']:
    #     yaml_file = "config/base_cfg.yaml"
    if method in ['mcnn']:
        yaml_file = "config/mcnn_cfg.yaml"
    elif method in ['stan']:
        yaml_file = "config/stan_cfg.yaml"
    elif method in ['stan_2d']:
        yaml_file = "config/stan_2d_cfg.yaml"
    elif method in ['stagn']:
        yaml_file = "config/stagn_cfg.yaml"
    elif method in ['gtan']:
        yaml_file = "config/gtan_cfg.yaml"
    elif method in ['rgtan']:
        yaml_file = "config/rgtan_cfg.yaml"
    elif method in ['hogrl']:
        yaml_file = "config/hogrl_cfg.yaml"
        
    else:
        raise NotImplementedError("Unsupported method.")

    # config = Config().get_config()
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    return args




def main(args):
    if args['method'] == 'mcnn':
        pass
    elif args['method'] == 'stan_2d':
        pass
    elif args['method'] == 'stan':
       pass
    elif args['method'] == 'stagn':
       pass
    elif args['method'] == 'gtan':
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
    main(parse_args())
