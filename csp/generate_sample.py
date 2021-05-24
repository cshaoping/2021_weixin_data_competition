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


def generate_sample(stage="offline_train"):
    """
    对负样本进行下采样，生成各个阶段所需样本
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    :return: List of sample df
    """
    day = STAGE_END_DAY[stage]
    if stage == "submit":
        sample_path = TEST_FILE
    else:
        sample_path = USER_ACTION
    stage_dir = os.path.join(ROOT_PATH, stage)
    df = pd.read_csv(sample_path)
    df_arr = []
    if stage == "evaluate":
        # 线下评估
        col = ["userid", "feedid", "date_", "device"] + ACTION_LIST
        df = df[df["date_"] == day][col]
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    elif stage == "submit":
        # 线上提交
        file_name = os.path.join(stage_dir, stage + "_" + "all" + "_" + str(day) + "_generate_sample.csv")
        df["date_"] = 15
        print('Save to: %s'%file_name)
        df.to_csv(file_name, index=False)
        df_arr.append(df)
    else:
        # 线下/线上训练
        # 同行为取按时间最近的样本
        for action in ACTION_LIST:
            df = df.drop_duplicates(subset=['userid', 'feedid', action], keep='last')
        # 负样本下采样
        for action in ACTION_LIST:
            action_df = df[(df["date_"] <= day) & (df["date_"] >= day - ACTION_DAY_NUM[action] + 1)]
            df_neg = action_df[action_df[action] == 0]
            df_neg = df_neg.sample(frac=1.0/ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
            df_all = pd.concat([df_neg, action_df[action_df[action] == 1]])
            col = ["userid", "feedid", "date_", "device"] + [action]
            file_name = os.path.join(stage_dir, stage + "_" + action + "_" + str(day) + "_generate_sample.csv")
            print('Save to: %s'%file_name)
            df_all[col].to_csv(file_name, index=False)
            df_arr.append(df_all[col])
    return df_arr


def concat_sample(sample_arr, stage="offline_train"):
    """
    基于样本数据和特征，生成特征数据
    :param sample_arr: List of sample df
    :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
    """
    day = STAGE_END_DAY[stage]
    # feed信息表
    feed_info = pd.read_csv(FEED_INFO)
    feed_info = feed_info.set_index('feedid')
    # 基于userid统计的历史行为的次数
    user_date_feature_path = os.path.join(ROOT_PATH, "feature", "userid_feature.csv")
    user_date_feature = pd.read_csv(user_date_feature_path)
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    user_date_feature.rename(columns={"read_commentsum": "read_commentsum_u", "likesum": "likesum_u", "click_avatarsum": "click_avatarsum_u",
                                      "forwardsum": "forwardsum_u", "commentsum": "commentsum_u", "followsum": "followsum_u",
                                      "favoritesum": "favoritesum_u"}, inplace=True)
    # 基于feedid统计的历史行为的次数
    feed_date_feature_path = os.path.join(ROOT_PATH, "feature", "feedid_feature.csv")
    feed_date_feature = pd.read_csv(feed_date_feature_path)
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])
    feed_date_feature.rename(columns={"read_commentsum": "read_commentsum_f", "likesum": "likesum_f", "click_avatarsum": "click_avatarsum_f",
                                      "forwardsum": "forwardsum_f", "commentsum": "commentsum_f", "followsum": "followsum_f",
                                      "favoritesum": "favoritesum_f"}, inplace=True)
    # 基于userid-feed_cate交叉统计特征
    user_author_feature_path = os.path.join(ROOT_PATH, "feature", "userid_authorid_feature.csv")
    user_author_feature = pd.read_csv(user_author_feature_path)
    user_author_feature = user_author_feature.set_index(["userid", "authorid", "date_"])
    user_author_feature.rename(columns={"read_commentsum": "read_commentsum_ua", "likesum": "likesum_ua", "click_avatarsum": "click_avatarsum_ua",
                                        "forwardsum": "forwardsum_ua", "commentsum": "commentsum_ua", "followsum": "followsum_ua",
                                        "favoritesum": "favoritesum_ua"}, inplace=True)

    user_bgm_song_feature_path = os.path.join(ROOT_PATH, "feature", "userid_bgm_song_id_feature.csv")
    user_bgm_song_feature = pd.read_csv(user_bgm_song_feature_path)
    user_bgm_song_feature = user_bgm_song_feature.set_index(["userid", "bgm_song_id", "date_"])
    user_bgm_song_feature.rename(columns={"read_commentsum": "read_commentsum_usong", "likesum": "likesum_usong", "click_avatarsum": "click_avatarsum_usong",
                                          "forwardsum": "forwardsum_usong", "commentsum": "commentsum_usong", "followsum": "followsum_usong",
                                          "favoritesum": "favoritesum_usong"}, inplace=True)

    user_bgm_singer_feature_path = os.path.join(ROOT_PATH, "feature", "userid_bgm_singer_id_feature.csv")
    user_bgm_singer_feature = pd.read_csv(user_bgm_singer_feature_path)
    user_bgm_singer_feature = user_bgm_singer_feature.set_index(["userid", "bgm_singer_id", "date_"])
    user_bgm_singer_feature.rename(columns={"read_commentsum": "read_commentsum_using", "likesum": "likesum_using", "click_avatarsum": "click_avatarsum_using",
                                            "forwardsum": "forwardsum_using", "commentsum": "commentsum_using", "followsum": "followsum_using",
                                            "favoritesum": "favoritesum_using"}, inplace=True)

    for index, sample in enumerate(sample_arr):
        features = ["userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id",
                    "videoplayseconds"]
        if stage == "evaluate":
            action = "all"
            features += ACTION_LIST
        elif stage == "submit":
            action = "all"
        else:
            action = ACTION_LIST[index]
            features += [action]
        print("action: ", action)
        sample = sample.join(feed_info, on="feedid", how="left", rsuffix="_feed")
        sample = sample.join(feed_date_feature, on=["feedid", "date_"], how="left", rsuffix="_feed")
        sample = sample.join(user_date_feature, on=["userid", "date_"], how="left", rsuffix="_user")
        sample = sample.join(user_author_feature, on=["userid", "authorid", "date_"], how="left", rsuffix="_ua")
        sample = sample.join(user_bgm_song_feature, on=["userid", "bgm_song_id", "date_"], how="left", rsuffix="_usong")
        sample = sample.join(user_bgm_singer_feature, on=["userid", "bgm_singer_id", "date_"], how="left", rsuffix="_usong")

        feed_feature_col = [b + "sum_f" for b in FEA_COLUMN_LIST]
        user_feature_col = [b + "sum_u" for b in FEA_COLUMN_LIST]
        user_author_feature_col = [b + "sum_ua" for b in FEA_COLUMN_LIST]
        user_bgm_song_feature_col = [b + "sum_usong" for b in FEA_COLUMN_LIST]
        user_bgm_singer_feature_col = [b + "sum_using" for b in FEA_COLUMN_LIST]

        sample[feed_feature_col] = sample[feed_feature_col].fillna(0.0)
        sample[user_feature_col] = sample[user_feature_col].fillna(0.0)
        sample[user_author_feature_col] = sample[user_author_feature_col].fillna(0.0)
        sample[user_bgm_song_feature_col] = sample[user_bgm_song_feature_col].fillna(0.0)
        sample[user_bgm_singer_feature_col] = sample[user_bgm_singer_feature_col].fillna(0.0)

        sample[feed_feature_col] = np.log(sample[feed_feature_col] + 1.0)
        sample[user_feature_col] = np.log(sample[user_feature_col] + 1.0)
        sample[user_author_feature_col] = np.log(sample[user_author_feature_col] + 1.0)
        sample[user_bgm_song_feature_col] = np.log(sample[user_bgm_song_feature_col] + 1.0)
        sample[user_bgm_singer_feature_col] = np.log(sample[user_bgm_singer_feature_col] + 1.0)

        features += feed_feature_col
        features += user_feature_col
        features += user_author_feature_col
        features += user_bgm_song_feature_col
        features += user_bgm_singer_feature_col

        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
        sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
            sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
        sample["videoplayseconds"] = np.log(sample["videoplayseconds"] + 1.0)

        sample[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
            sample[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
        file_name = os.path.join(ROOT_PATH, stage, stage + "_" + action + "_" + str(day) + "_concate_sample.csv")
        print('Save to: %s' % file_name)
        sample[features].to_csv(file_name, index=False)


if __name__ == "__main__":
    t = time.time()
    for stage in STAGE_END_DAY:
        logger.info("Stage: %s" % stage)
        logger.info("Generate sample")
        sample_arr = generate_sample(stage)
        logger.info('Concat sample with feature')
        concat_sample(sample_arr, stage)
    print('Time cost: %.2f s' % (time.time()-t))
