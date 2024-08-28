import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
import warnings
from util import Logger

import itertools
from sklearn.preprocessing import LabelEncoder
import re
import datetime

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.safe_load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']  # RAWデータ格納場所
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME'] # モデルの格納場所
if 'REMOVE_COLS' in yml['SETTING'].keys():
    REMOVE_COLS = yml['SETTING']['REMOVE_COLS']


#### preprocessing関数を定義 ##########################################################

# all_dfを受け取りall_dfを返す関数



def prep_price(df):

    df["Price"] = df["Price"].apply(np.log1p)

    return df

def prep_label_encoding(df):

    # カテゴリ変数のカラムを抽出
    obj_cats = df.dtypes[df.dtypes=="object"].index

    # 
    for cat in obj_cats:
        le = LabelEncoder().fit(df[cat])
        df[cat] = le.transform(df[cat])

    return df

def prep_nan(df):
    """欠損値の補完、ニューラルネット用
    """

    df = df.fillna(0)

    return df



##### main関数を定義 ###########################################################################
def main():
    
    # データの読み込み
    train = pd.read_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test = pd.read_pickle(FEATURE_DIR_NAME + 'test.pkl')
    df = pd.concat([train, test], ignore_index=True)
    print("train shape: ", train.shape)
    print("test shape: ", test.shape)
    
    # 前処理
    df = prep_label_encoding(df)
    nn_df = prep_nan(df)
    
    # pickleファイルとして保存
    vs_train = df.iloc[:len(train), :]
    vs_test = df.iloc[len(train):, :]
    vs_train.to_pickle(FEATURE_DIR_NAME + 'vs_train.pkl')
    vs_test.to_pickle(FEATURE_DIR_NAME + 'vs_test.pkl')

    nn_train = nn_df.iloc[:len(train), :]
    nn_test = nn_df.iloc[len(train):, :]
    nn_train.to_pickle(FEATURE_DIR_NAME + 'nn_train.pkl')
    nn_test.to_pickle(FEATURE_DIR_NAME + 'nn_test.pkl')
#     logger.info(f'train shape: {train.shape}, test shape, {test.shape}')
    
    # 生成した特徴量のリスト
    features_list = list(df.columns)
    if 'REMOVE_COLS' in yml['SETTING'].keys():
         # 学習に不要なカラムは除外
        features_list = list(df.drop(columns=REMOVE_COLS).columns) 
    
    # 特徴量リストの保存
    # features_list = sorted(features_list)
    with open('../features_list.txt', 'wt', encoding='utf-8') as f:
        for i in range(len(features_list)):
            f.write('\'' + str(features_list[i]) + '\',\n')
    
    return 'main() Done!'

    
if __name__ == '__main__':
    
#     global logger
#     logger = Logger(MODEL_DIR_NAME + "create_features" + "/")

    message = main()
    print(message)
