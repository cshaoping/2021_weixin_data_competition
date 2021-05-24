# coding: utf-8
import os
import time
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd
from const import *


# 生成单维度的统计特征
def create_statistic_feature(start_day=1, before_day=7, agg="sum"):
    history_data = pd.read_csv(USER_ACTION, nrows=100)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feature_dir = os.path.join(ROOT_PATH, "feature")
    for dim in ["userid", "feedid"]:
        print(dim)
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        for start in range(start_day, END_DAY-before_day+1):
            temp = user_data[(user_data["date_"]) >= start & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg([agg]).reset_index()
            temp.columns = list(map(''.join, temp.columns.values))
            temp["date_"] = start + before_day
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim+"_feature.csv")
        print('Save to: %s'%feature_path)
        dim_feature.to_csv(feature_path, index=False)


# 生成交叉维度的统计特征
def create_cross_statistic_feature(start_day=1, before_day=7, agg="sum"):
    feed_cate_list = ["authorid", "bgm_song_id", "bgm_singer_id"]
    history_data = pd.read_csv(USER_ACTION, nrows=1000)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feed_info = pd.read_csv(FEED_INFO, nrows=1000)[["feedid"] + feed_cate_list]
    feature_dir = os.path.join(ROOT_PATH, "feature")
    history_data = pd.merge(history_data, feed_info, how='inner', on='feedid')
    for cate_name in feed_cate_list:
        print(cate_name)
        user_data = history_data[["userid", cate_name, "date_"] + FEA_COLUMN_LIST]
        user_data = user_data[history_data[cate_name] > 0]
        res_arr = []
        for start in range(start_day, END_DAY - before_day + 1):
            temp = user_data[(user_data["date_"]) >= start & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby(["userid", cate_name]).agg([agg]).reset_index()
            temp.columns = list(map(''.join, temp.columns.values))
            temp["date_"] = start + before_day
            res_arr.append(temp)
        dim_feature = pd.concat(res_arr)
        dim_feature[cate_name] = dim_feature[cate_name].values.astype("int64")
        feature_path = os.path.join(feature_dir, "userid_" + cate_name + "_feature.csv")
        print('Save to: %s' % feature_path)
        dim_feature.to_csv(feature_path, index=False)


if __name__ == "__main__":
    t = time.time()
    logger.info("Create statistic feature")
    # userid 和 feedid单维度的统计特征
    create_statistic_feature()
    # userid 和 feed相关属性特征的交叉统计特征
    create_cross_statistic_feature()

    print("Time cost: %.2f s" % (time.time() - t))
