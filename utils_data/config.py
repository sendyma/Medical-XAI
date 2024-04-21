import os
import yaml
from easydict import EasyDict
import argparse


def optionFlags():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config_file', help='Config file for the environment')
    args = parser.parse_args()
    return args


def create_config(config_file_exp):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = EasyDict()
    # Copy
    for k, v in config.items():
        cfg[k] = v
    return cfg 


