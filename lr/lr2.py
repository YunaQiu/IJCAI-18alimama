#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： SGDClassifier
模型参数：loss="log",
        learning_rate='optimal',
        n_iter=500,
        n_jobs=-1,
        random_state=0
数值特征：小时转化率，商品历史转化率，店铺历史转化率
onehot特征：
    商品销量等级，商品收藏量等级，商品价格等级，商品一级类目，商品广告等级，商品二级类目，商品所在城市，品牌id，商品id
    用户性别编号，用户年龄段，用户星级等级，用户职业编号
    展示页码编号
    店铺服务评分,店铺物流评分,店铺id

    上下文预测类目与商品类目交集个数，上下文属性与商品属性交集个数，
    用户距离上次点击时长（秒），用户过去一小时点击量，用户前一天点击量，用户前一天转化率，用户历史转化率，用户里是点击数
    用户距离下次点击时长，用户未来一小时点击量，用户前后两次点击时长差值，用户前后两小时点击次数差值
    用户年龄与类目年龄差值
    商品历史转化率（当天以前），商品前一天交易量，商品前一天转化率，商品前一天交易量与同类差值，商品前一天转化率与同类差值
    用户年龄与商品平均用户年龄差值，商品销售等级与同类之差，商品收藏与同类之差，商品广告与同类之差
    商品属性个数，
    店铺前一天交易量，店铺前一天转化率，店铺前一天点击量与同类差值
    用户年龄与店铺平均用户年龄差值
    店铺商品数目
    店铺平均销量与同类之差，店铺平均收藏与同类之差，店铺平均广告投放与同类之差
    品牌历史转化率，品牌前一天转化率，
    品牌平均收藏与同类差值，品牌平均价格与同类差值，品牌平均广告与同类差值，品牌平均销量与同类差值，
    用户年龄与品牌平均年龄之差，
    店铺该品牌商品占有率，店铺品牌数与商品数比值

    用户在该类目的历史转化率，用户距离下次点击该类目时长，用户前后两次点击该类目时间之差，用户在该类目前后两小时点击数差值，
    用户在该类目平均价格，商品价格与用户在该类目平均价格之差
    用户距离上次点击该商品时长，用户过去一小时点击该商品次数，用户前一天点击该商品次数，用户过去一小时点击该商品与点击该类目次数的比例，
    类目在该年龄段的转化率，类目在该性别的转化率
    商品在该年龄段的转化率，商品在该年龄段的转化率与同类之差，商品在该性别的转化率，商品在该性别的转化率与同类之差
前期处理：对数值特征进行归一化缩放处理，对onehot特征的特征值进行分区，对个别特征数量过少的分区进行合并
结果： B榜（0.14143）

'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import urllib, urllib.parse, urllib.request
import json, random

from sklearn.preprocessing import *
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep=' ', na_values='-1', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, na_values='-1', header=header, index_col=index_col, names=colNames)
    return df

# 添加one-hot编码并保留原字段
def addOneHot(df, colName):
    if isinstance(colName, str):
        colName = [colName]
    colTemp = df[colName]
    df = pd.get_dummies(df, columns=colName)
    df = pd.concat([df, colTemp], axis=1)
    return df

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True, subset=cols)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler

# 对数组集合进行合并操作
def listAdd(l):
    result = []
    [result.extend(x) for x in l]
    return result

# 对不同标签进行抽样处理
def getSubsample(labelList, ratio=0.8, repeat=False, params=None):
    if not isinstance(params, dict):
        if isinstance(ratio, (float, int)):
            params = {k:{'ratio':ratio, 'repeat':repeat} for k in set(labelList)}
        else:
            params={k:{'ratio':ratio[k], 'repeat':repeat} for k in ratio.keys()}
    resultIdx = []
    for label in params.keys():
        param = params[label]
        tempList = np.where(labelList==label)[0]
        sampleSize = np.ceil(len(tempList)*params[label]['ratio']).astype(int)
        if (~param['repeat'])&(param['ratio']<=1):
            resultIdx.extend(random.sample(tempList.tolist(),sampleSize))
        else:
            resultIdx.extend(tempList[np.random.randint(len(tempList),size=sampleSize)])
    return resultIdx

# 矩估计法计算贝叶斯平滑参数
def countBetaParamByMME(inputArr):
    EX = inputArr.mean()
    EX2 = (inputArr ** 2).mean()
    alpha = (EX*(EX-EX2)) / (EX2 - EX**2)
    beta = alpha * (1/EX - 1)
    return alpha,beta

# 对numpy数组进行贝叶斯平滑处理
def biasSmooth(aArr, bArr, method='MME', alpha=None, beta=None):
    ratioArr = aArr / bArr
    if method=='MME':
        alpha,beta = countBetaParamByMME(ratioArr[ratioArr==ratioArr])
    resultArr = (aArr+alpha) / (bArr+alpha+beta)
    return resultArr

# 添加商品类别统计特征
def addCateFea(df, originDf=None, **params):
    df['cate_age_delta_level'] = df['cate_age_delta'] // 1
    df['cate_age_delta_level'].fillna(-9999,inplace=True)
    return df

# 添加商品历史浏览量及购买量特征
def addShopFea(df, originDf=None, **params):
    df['shop_review_positive_rate_level'] = df['shop_review_positive_rate'] // 0.005
    df['shop_score_service_level'] = df['shop_score_service'] // 0.005
    df['shop_score_delivery_level'] = df['shop_score_delivery'] // 0.005
    df['shop_score_description_level'] = df['shop_score_description'] // 0.005
    df.loc[df.shop_review_positive_rate_level<186, 'shop_review_positive_rate_level'] = 185
    df.loc[df.shop_score_service_level<188, 'shop_score_service_level'] = 187
    df.loc[df.shop_score_delivery_level<184, 'shop_score_delivery_level'] = 183
    df.loc[df.shop_score_description_level<186, 'shop_score_description_level'] = 185

    df['shop_lastdate_trade_level'] = df['shop_lastdate_trade'] // 2
    # df.loc[df.shop_lastdate_trade_level>8,'shop_lastdate_trade_level'] = 9
    df['shop_lastdate_trade_level'].fillna(-1, inplace=True)
    df['shop_lastdate_trade_ratio_level'] = df['shop_lastdate_trade_ratio'] // 0.005
    # df.loc[df.shop_lastdate_trade_ratio_level>11,'shop_lastdate_trade_ratio_level'] = 12
    df['shop_lastdate_trade_ratio_level'].fillna(-1, inplace=True)
    df['shop_lastdate_show_delta_level'] = df['shop_lastdate_show_delta'] // 1000
    # df.loc[df.shop_lastdate_show_delta_level>7,'shop_lastdate_show_delta_level'] = 8
    df['shop_lastdate_show_delta_level'].fillna(-1, inplace=True)

    df['shop_age_delta_level'] = df['shop_age_delta'] // 1
    # df.loc[df.shop_age_delta_level>3,'shop_age_delta_level'] = 3
    # df.loc[df.shop_age_delta_level<-2,'shop_age_delta_level'] = -3
    df.loc[df.shop_age_delta_level.isnull(),'shop_age_delta_level'] = -9999

    df['shop_sales_delta_level'] = df['shop_sales_delta'] // 1
    # df.loc[df.shop_sales_delta_level>6,'shop_sales_delta_level'] = 7
    # df.loc[df.shop_sales_delta_level<-6,'shop_sales_delta_level'] = -6
    df['shop_collected_delta_level'] = df['shop_collected_delta'] // 1
    # df.loc[df.shop_collected_delta_level>6,'shop_collected_delta_level'] = 7
    # df.loc[df.shop_collected_delta_level<-5,'shop_collected_delta_level'] = -6
    df['shop_pv_delta_level'] = df['shop_pv_delta'] // 1
    # df.loc[df.shop_pv_delta_level>6,'shop_pv_delta_level'] = 7
    # df.loc[df.shop_pv_delta_level<-4,'shop_pv_delta_level'] = -5
    df['shop_item_count_level'] = df['shop_item_count'] // 5
    # df.loc[df.shop_item_count_level>15,'shop_item_count_level'] = 15

    df['shop_review_num_delta_level'] = df['shop_review_num_delta'] // 2
    # df.loc[df.shop_review_num_delta_level>4,'shop_review_num_delta_level'] = 4
    # df.loc[df.shop_review_num_delta_level<-3,'shop_review_num_delta_level'] = -3
    df['shop_star_level_delta_level'] = df['shop_star_level_delta'] // 2
    # df.loc[df.shop_star_level_delta_level>3,'shop_star_level_delta_level'] = 3
    # df.loc[df.shop_star_level_delta_level<-2,'shop_star_level_delta_level'] = -2

    df['shop_brand_item_ratio_level'] = df['shop_brand_item_ratio'] // 0.1
    # df.loc[df.shop_brand_item_ratio_level<3,'shop_brand_item_ratio_level'] = 3
    df.loc[df.shop_brand_item_ratio_level.isnull(),'shop_brand_item_ratio_level'] = -1
    df['shop_brand_count_ratio_level'] = df['shop_brand_count_ratio'] // 0.1
    # df.loc[df.shop_brand_count_ratio_level>3,'shop_brand_count_ratio_level'] = 3
    df.loc[df.shop_brand_count_ratio_level.isnull(),'shop_brand_count_ratio_level'] = -1
    df['shop_brand_special_degree_level'] = df['shop_brand_special_degree'] // 0.05
    # df.loc[df.shop_brand_special_degree_level>3,'shop_brand_special_degree_level'] = 3
    df.loc[df.shop_brand_special_degree_level.isnull(),'shop_brand_special_degree_level'] = -1
    return df

# 添加商品历史浏览量及购买量特征
def addItemFea(df, originDf=None, **params):
    df['item_prop_num_level'] = df['item_prop_num'] // 5
    df['item_his_trade_level'] = df['item_his_trade'] // 10
    df['item_lastdate_trade_level'] = df['item_lastdate_trade'] // 0.005
    df['item_lastdate_trade_level'].fillna(-9999, inplace=True)
    df['item_lastdate_trade_ratio_level'] = df['item_lastdate_trade_ratio'] // 0.005
    df['item_lastdate_trade_ratio_level'].fillna(-9999, inplace=True)

    df['item_lastdate_trade_delta_level'] = df['item_lastdate_trade_delta'] // 1
    df['item_lastdate_trade_delta_level'].fillna(-9999, inplace=True)
    df['item_lastdate_trade_ratio_delta_level'] = df['item_lastdate_trade_ratio_delta'] // 0.01
    df['item_lastdate_trade_ratio_delta_level'].fillna(-9999, inplace=True)

    df['item_sales_delta_level'] = df['item_sales_delta'] // 1
    df['item_collected_delta_level'] = df['item_collected_delta'] // 1
    df['item_pv_delta_level'] = df['item_pv_delta'] // 1
    df['item_age_delta_level'] = df['item_age_delta'] // 1
    df['item_age_delta_level'].fillna(-9999, inplace=True)
    return df

# 添加用户维度特征
def addUserFea(df, originDf=None, **params):
    df['user_last_show_timedelta_level'] = df['user_last_show_timedelta'] // 100
    # df.loc[(df.user_last_show_timedelta_level>14)&(df.user_last_show_timedelta_level<999999),'user_last_show_timedelta_level'] = 15
    df['user_next_show_timedelta_level'] = df['user_next_show_timedelta'] // 100
    # df.loc[(df.user_next_show_timedelta_level>13)&(df.user_next_show_timedelta_level<999999),'user_next_show_timedelta_level'] = 14
    df['user_his_show_level'] = df['user_his_show'].values
    # df.loc[df.user_his_show_level>23,'user_his_show_level'] = 24
    df['user_lasthour_show_level'] = df['user_lasthour_show'].values
    # df.loc[df.user_lasthour_show_level>17,'user_lasthour_show_level'] = 18
    df['user_nexthour_show_level'] = df['user_nexthour_show'].values
    # df.loc[df.user_nexthour_show_level>13,'user_nexthour_show_level'] = 14
    df['user_lastdate_show_level'] = df['user_lastdate_show'].values
    # df.loc[df.user_lastdate_show_level>22,'user_lastdate_show_level'] = 23
    df['user_lastdate_trade_ratio_level'] = df['user_lastdate_trade_ratio'] // 0.0005
    # df.loc[df.user_lastdate_trade_ratio_level>20,'user_lastdate_trade_ratio_level'] = 21
    df['user_his_trade_ratio_level'] = df['user_his_trade_ratio'] // 0.0005
    # df.loc[df.user_his_trade_ratio_level>19,'user_his_trade_ratio_level'] = 20
    df['user_near_timedelta_level'] = df['user_near_timedelta'] // 600
    # df.loc[df.user_near_timedelta_level<0,'user_near_timedelta_level'] = -1
    # df.loc[df.user_near_timedelta_level>0,'user_near_timedelta_level'] = 1
    df.loc[df.user_near_timedelta_level.isnull(),'user_his_trade_ratio_level'] = 999999
    df['user_nearhour_show_delta_level'] = df['user_nearhour_show_delta'] // 600
    # df.loc[df.user_nearhour_show_delta_level>7,'user_nearhour_show_delta_level'] = 8
    # df.loc[df.user_nearhour_show_delta_level<-8,'user_nearhour_show_delta_level'] = -9
    df.loc[df.user_nearhour_show_delta_level.isnull(),'user_nearhour_show_delta_level'] = 999999
    return df

# 添加广告商品与查询词的相关性特征
def addContextFea(df, **params):
    df['predict_cate_num_level'] = df['predict_cate_num'].values
    df['predict_prop_num_level'] = df['predict_prop_num'].values
    df['prop_intersect_num_level'] = df['prop_intersect_num'].values
    # df.loc[df.predict_cate_num_level>10, 'predict_cate_num_level'] = 11
    # df.loc[df.predict_prop_num_level>8, 'predict_prop_num_level'] = 9
    # df.loc[df.prop_intersect_num_level>4, 'prop_intersect_num_level'] = 5

    df['prop_jaccard_level'] = df['prop_jaccard'] // 0.01
    df.loc[df.prop_jaccard_level>11, 'prop_jaccard_level'] = 12
    df['prop_jaccard_bias_level'] = df['prop_jaccard_bias'] // 0.005
    df.loc[df.prop_jaccard_bias_level>13, 'prop_jaccard_bias_level'] = 14

    df.fillna({k:-1 for k in ['predict_cate_num_level','predict_prop_num_level','prop_intersect_num_level','prop_jaccard_level','prop_jaccard_bias_level']})
    return df

# 添加品牌相关特征
def addBrandFea(df, originDf=None, **params):
    df['brand_his_trade_ratio_level'] = df['brand_his_trade_ratio'] // 0.01
    # df.loc[df.brand_his_trade_ratio_level>6,'brand_his_trade_ratio_level'] = 6
    df.loc[df.brand_his_trade_ratio_level.isnull(),'brand_his_trade_ratio_level'] = -1
    df['brand_lastdate_trade_ratio_level'] = df['brand_lastdate_trade_ratio'] // 0.01
    # df.loc[df.brand_lastdate_trade_ratio_level>5,'brand_lastdate_trade_ratio_level'] = 5
    df.loc[df.brand_lastdate_trade_ratio_level.isnull(),'brand_lastdate_trade_ratio_level'] = -1
    df['brand_collected_delta_level'] = df['brand_collected_delta'] // 1
    # df.loc[df.brand_collected_delta_level>6,'brand_collected_delta_level'] = 6
    # df.loc[df.brand_collected_delta_level<-5,'brand_collected_delta_level'] = -5
    df.loc[df.brand_collected_delta_level.isnull(),'brand_collected_delta_level'] = -9999
    df['brand_price_delta_level'] = df['brand_price_delta'] // 1
    # df.loc[df.brand_price_delta_level<-2,'brand_price_delta_level'] = -2
    # df.loc[df.brand_price_delta_level>1,'brand_price_delta_level'] = 1
    df.loc[df.brand_price_delta_level.isnull(),'brand_price_delta_level'] = -9999
    df['brand_pv_delta_level'] = df['brand_pv_delta'] // 1
    # df.loc[df.brand_pv_delta_level<-3,'brand_pv_delta_level'] = -3
    # df.loc[df.brand_pv_delta_level>6,'brand_pv_delta_level'] = 6
    df.loc[df.brand_pv_delta_level.isnull(),'brand_pv_delta_level'] = -9999
    df['brand_sales_delta_level'] = df['brand_sales_delta'] // 1
    # df.loc[df.brand_sales_delta_level<-5,'brand_sales_delta_level'] = -5
    # df.loc[df.brand_sales_delta_level>6,'brand_sales_delta_level'] = 6
    df.loc[df.brand_sales_delta_level.isnull(),'brand_sales_delta_level'] = -9999
    df['brand_age_delta_level'] = df['brand_age_delta'] // 1
    # df.loc[df.brand_age_delta_level<-3,'brand_age_delta_level'] = -3
    # df.loc[df.brand_age_delta_level>2,'brand_age_delta_level'] = 2
    df.loc[df.brand_age_delta_level.isnull(),'brand_age_delta_level'] = -9999
    return df

# 添加用户与类目关联维度的特征
def addUserCateFea(df, originDf=None, **params):
    df['uc_his_trade_ratio_level'] = df['uc_his_trade_ratio'] // 0.0001
    # df.loc[df.uc_his_trade_ratio_level>250,'uc_his_trade_ratio_level'] = 251
    # df.loc[df.uc_his_trade_ratio_level<3,'uc_his_trade_ratio_level'] = 2
    df['uc_next_show_timedelta_level'] = df['uc_his_trade_ratio'] // 0.0001
    # df.loc[(df.uc_next_show_timedelta_level>5)&(df.uc_next_show_timedelta_level<999999),'uc_next_show_timedelta_level'] = 6
    df['uc_price_mean_level'] = df['uc_price_mean'] // 1
    # df.loc[df.uc_price_mean_level<4,'uc_price_mean_level'] = 4
    # df.loc[df.uc_price_mean_level>9,'uc_price_mean_level'] = 9
    df['uc_price_delta_level'] = df['uc_price_delta'] // 1
    # df.loc[df.uc_price_delta_level<-2,'uc_price_delta_level'] = -2
    # df.loc[df.uc_price_delta_level>2,'uc_price_delta_level'] = 2
    df['uc_near_timedelta_level'] = df['uc_near_timedelta'] // 100
    # df.loc[df.uc_near_timedelta_level>10,'uc_near_timedelta_level'] = 10
    # df.loc[df.uc_near_timedelta_level<-10,'uc_near_timedelta_level'] = -10
    # df.loc[(df.uc_near_timedelta_level>1)&(df.uc_near_timedelta_level<10),'uc_near_timedelta_level'] = 5
    # df.loc[(df.uc_near_timedelta_level<-1)&(df.uc_near_timedelta_level>-10),'uc_near_timedelta_level'] = -5
    df.loc[df.uc_near_timedelta_level.isnull(),'uc_near_timedelta_level'] = 999999
    df['uc_nearhour_show_delta_level'] = df['uc_nearhour_show_delta'].values
    # df.loc[df.uc_nearhour_show_delta_level>4, 'uc_nearhour_show_delta_level'] = 5
    # df.loc[df.uc_nearhour_show_delta_level<-4, 'uc_nearhour_show_delta_level'] = -5
    df.loc[df.uc_nearhour_show_delta_level.isnull(),'uc_nearhour_show_delta_level'] = 999999
    return df

# 添加用户商品关联维度的特征
def addUserItemFea(df, originDf=None, **params):
    df['ui_last_show_timedelta_level'] = df['ui_last_show_timedelta'] // 300
    # df.loc[(df.ui_last_show_timedelta_level>3)&(df.ui_last_show_timedelta_level<99999),'ui_last_show_timedelta_level'] = 3
    df['ui_lasthour_show_level'] = df['ui_lasthour_show'].values
    # df.loc[df.ui_lasthour_show_level>2,'ui_lasthour_show_level'] = 2
    df['ui_lasthour_show_ratio_level'] = df['ui_lasthour_show_ratio'] // 0.1
    # df.loc[df.ui_lasthour_show_ratio_level>4,'ui_lasthour_show_ratio_level'] = 5
    df['ui_lastdate_show_level'] = df['ui_lastdate_show'].values
    # df.loc[df.ui_lastdate_show_level>2,'ui_lastdate_show_level'] = 2
    return df

# 统计用户该价格段商品的统计特征
def addUserShopFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','shop_id'], **params['statDateTrade'])
    tempDf.columns = ['us_his_show','us_his_trade']
    tempDf['us_his_trade_ratio'] = biasSmooth(tempDf['us_his_trade'].values, tempDf['us_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','shop_id','date'], right_index=True)
    return df

# 统计用户该价格段商品的统计特征
def addUserPriceFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','item_price_level'], **params['statDateTrade'])
    tempDf.columns = ['up_his_show','up_his_trade']
    tempDf['up_his_trade_ratio'] = biasSmooth(tempDf['up_his_trade'].values, tempDf['up_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','item_price_level','date'], right_index=True)
    for x in set(df.item_price_level.dropna().values):
        idx = df[df.item_price_level==x].index
        df.loc[idx, 'up_his_show_ratio'] = biasSmooth(df.loc[idx, 'up_his_show'].values, df.loc[idx,'user_his_show'].values)
    return df

# 统计用户该价格段商品的统计特征
def addUserHourFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','hour'], **params['statDateTrade'])
    tempDf.columns = ['uh_his_show','uh_his_trade']
    tempDf['uh_his_trade_ratio'] = biasSmooth(tempDf['uh_his_trade'].values, tempDf['uh_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['user_id','hour','date'], right_index=True)
    for x in set(df.hour.dropna().values):
        idx = df[df.hour==x].index
        df.loc[idx, 'uh_his_show_ratio'] = biasSmooth(df.loc[idx, 'uh_his_show'].values, df.loc[idx,'user_his_show'].values)
    return df

# 统计商品与各类别用户群交叉特征
def addCateCrossFea(df, **params):
    df['ca_his_trade_ratio'] = df['ca_his_trade_ratio'] // 0.005
    # df.loc[df.ca_his_trade_ratio>8,'ca_his_trade_ratio'] = 8
    df.loc[df.ca_his_trade_ratio.isnull(),'ca_his_trade_ratio'] = -999999
    df['cg_his_trade_ratio'] = df['cg_his_trade_ratio'] // 0.005
    # df.loc[df.cg_his_trade_ratio>7,'cg_his_trade_ratio'] = 7
    return df

# 统计商品与各类别用户群交叉特征
def addItemCrossFea(df, **params):
    df['ia_his_trade_ratio_level'] = df['ia_his_trade_ratio'] // 0.005
    # df.loc[df.ia_his_trade_ratio_level>9,'ia_his_trade_ratio_level'] = 10
    df.loc[df.ia_his_trade_ratio_level.isnull(),'ia_his_trade_ratio_level'] = -999999
    df['ia_his_trade_delta_level'] = df['ia_his_trade_delta'] // 0.01
    # df.loc[(df.ia_his_trade_delta_level<-4),'ia_his_trade_delta_level'] = -4
    # df.loc[(df.ia_his_trade_delta_level>5),'ia_his_trade_delta_level'] = 5
    df.loc[df.ia_his_trade_delta_level.isnull(),'ia_his_trade_delta_level'] = -999999
    df['ig_his_trade_ratio_level'] = df['ig_his_trade_ratio'] // 0.005
    # df.loc[df.ig_his_trade_ratio_level>14,'ig_his_trade_ratio_level'] = 14
    df['ig_his_trade_delta_level'] = df['ig_his_trade_delta'] // 0.01
    # df.loc[df.ig_his_trade_delta_level>3,'ig_his_trade_delta_level'] = 4
    # df.loc[df.ig_his_trade_delta_level<-4,'ig_his_trade_delta_level'] = -4
    return df

# 统计品牌与各类别用户群交叉特征
def addBrandCrossFea(df, **params):
    tempDf = statDateTrade(df, ['item_brand_id','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ba_his_show','ba_his_trade']
    tempDf['ba_his_trade_ratio'] = biasSmooth(tempDf['ba_his_trade'].values, tempDf['ba_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_age_level','date'], right_index=True)

    tempDf = statDateTrade(df, ['item_brand_id','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['bg_his_show','bg_his_trade']
    tempDf['bg_his_trade_ratio'] = biasSmooth(tempDf['bg_his_trade'].values, tempDf['bg_his_show'].values)
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_gender_id','date'], right_index=True)
    return df

# 数据清洗
def dataCleaning(df):
    startTime = datetime.now()
    df['context_timestamp'] = df['timestamp'].map(lambda x: datetime.fromtimestamp(x))
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_property_list_str']]
    tempDf['item_property_list'] = tempDf[tempDf.item_property_list_str.notnull()]['item_property_list_str'].map(lambda x: x.split(';'))
    df = df.merge(tempDf, how='left', on='item_id')
    df['predict_category_property'] = df[df.predict_category_property_str.notnull()]['predict_category_property_str'].map(
        lambda x: {kv.split(':')[0]:(kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) for kv in x.split(';')})
    df['date'] = pd.to_datetime(df['date_str'])
    print('cleaning time:', datetime.now()-startTime)
    return df

# 特征方法汇总
def feaFactory(df, originDf=None, **args):
    params = {
        'statDateTrade': {
            'statDates': None,
            'skipDates': 1,
        }
    }
    for k1,arg in args.items():
        for k2,v in arg.items():
            params[k1][k2] = v
    startTime = datetime.now()
    # df = addTimeFea(df, **params)
    df = addCateFea(df, originDf, **params)
    df = addUserFea(df, originDf, **params)
    df = addShopFea(df, originDf, **params)
    df = addItemFea(df, originDf, **params)
    df = addContextFea(df, **params)
    df = addBrandFea(df,originDf, **params)
    df = addUserCateFea(df, originDf, **params)
    df = addUserItemFea(df, originDf, **params)
    # df = addUserShopFea(df, originDf, **params)
    # df = addUserPriceFea(df, originDf, **params)
    # df = addUserHourFea(df, originDf, **params)
    df = addCateCrossFea(df, **params)
    # df = addBrandCrossFea(df, **params)
    df = addItemCrossFea(df, **params)
    print('finished feaFactory: ', datetime.now() - startTime)
    return df

# 字典矩阵转化
class DictDataset():
    def __init__(self, df, numFea=[], strFea=[]):
        self.numFea = numFea
        self.strFea = strFea
        print('fea counts: ', len(numFea) + len(strFea))

        if len(numFea)>0:
            df, scaler = scalerFea(df.fillna({k:0 for k in numFea}), numFea)
            self.scaler = scaler
        else:
            self.scaler = None
        dfV = self.getDictList(df)
        dv = DictVectorizer().fit(dfV)
        self.dv = dv

    def getDictList(self, df):
        numFea = self.numFea
        strFea = self.strFea

        df.loc[:,numFea] = df.loc[:,numFea].astype(float)
        df.loc[:,strFea] = df.loc[:,strFea].astype(str)
        df.loc[:,strFea] = df.loc[:,strFea].applymap(lambda x: np.nan if x=='nan' else x)
        if len(numFea)>0:
            df.fillna({k:0 for k in numFea}, inplace=True)
            df.loc[:,numFea] = self.scaler.transform(df[numFea].values)
        dfV = df.to_dict('records')
        for i,x in enumerate(dfV):
            dfV[i] = {k:v for k,v in x.items() if v==v}
        return dfV

    def transform(self, df):
        dfV = self.getDictList(df)
        return self.dv.transform(dfV)

class SGDModel():
    def train(self, X, y, verbose=0, save=False, modelName='sgd'):
        self.clf = linear_model.SGDClassifier(loss="log", verbose=verbose, learning_rate='optimal', eta0=0.0002, n_iter=500, n_jobs=-1, random_state=0)
        self.clf.fit(X,y)
        if save:
            joblib.dump(self.clf, './%s.pkl' % (modelName), compress=3)

    def predict(self, X):
        result = self.clf.predict_proba(X)
        result = result.T[1].T
        return result

    def gridSearch(self, X, y, nFold=4, verbose=1, num_boost_round=130):
        paramsGrids = {
            # 'alpha': [0.0001*i for i in range(1,10)],
            'n_iter': [100+50*i for i in range(0,10)],
        }
        for k,v in paramsGrids.items():
            gsearch = GridSearchCV(
                estimator = linear_model.SGDClassifier(loss="log", learning_rate='optimal', alpha=0.0002, eta0=0.9, n_iter=200, n_jobs=-1, random_state=0),
                # param_grid = paramsGrids,
                param_grid = {k:v},
                scoring = 'neg_log_loss',
                cv = nFold,
                verbose = verbose,
                n_jobs = 4
            )
            gsearch.fit(X, y)
            print(pd.DataFrame(gsearch.cv_results_))
            print(gsearch.best_params_)
        # exit()

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=5, stratify=False, weight=None):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    if stratify:
        kf = StratifiedKFold(n_splits=nFold, shuffle=True)
    else:
        kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX, trainY)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        if weight != None:
            kfWeight = weight[trainIdx]
        else:
            kfWeight = None
        clf.train(kfTrainX, kfTrainY)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
    oofTest[:] = oofTestSkf.mean(axis=1)
    return oofTrain, oofTest


if __name__ == '__main__':
    # 准备数据
    startTime = datetime.now()
    df = importDf('../data/train_fea_special_sample.csv')
    # df = importDf('../data/train_fea_special.csv')
    df['dataset'] = 0
    predictDf = df.sample(frac=0.1)
    # predictDf = importDf('../data/test_b_fea.csv')
    predictDf['dataset'] = -2
    originDf = pd.concat([df,predictDf], ignore_index=True)
    # originDf = originDf.sample(frac=0.5)
    print('prepare dataset time:', datetime.now()-startTime)

    # 特征处理
    startTime = datetime.now()
    originDf = dataCleaning(originDf)
    originDf = feaFactory(originDf)
    df = originDf[(originDf['dataset']>=0)]
    predictDf = originDf[originDf['dataset']==-2]

    # 特征筛选
    numFea = [
        'hour_trade_ratio',
        # 'user_his_show',
        # 'user_his_trade_ratio',
        'item_his_trade_ratio',
        'shop_his_trade_ratio',
        # 'cate_his_trade_ratio','cate2_his_trade_ratio',
        # 'uc_his_trade_ratio',
        # 'ca_his_trade_ratio','cg_his_trade_ratio',
        # 'ia_his_trade_ratio',
        # 'ig_his_trade_ratio',
    ]
    strFea = [
        'item_sales_level','item_price_level','item_collected_level','item_pv_level','item_city_id','item_category1','item_category2','item_brand_id','item_id',
        'user_gender_id','user_age_level','user_occupation_id','user_star_level',#'user_id',
        'context_page_id',#'hour',
        'shop_score_service_level','shop_score_delivery_level','shop_id',#'shop_score_description_level','shop_star_level','shop_review_positive_rate_level','shop_review_num_level',

        'cate_intersect_num','prop_intersect_num_level',#'prop_jaccard_level','prop_jaccard_bias_level','predict_cate_num_level','predict_prop_num_level',
        # 'predict_cate_num_level','cate_intersect_num','predict_prop_num_level','prop_intersect_num_level',#'prop_jaccard_level','prop_jaccard_bias_level',
        'user_last_show_timedelta_level','user_lasthour_show_level','user_lastdate_show_level','user_lastdate_trade_ratio_level','user_his_trade_ratio_level','user_his_show_level',
        'user_next_show_timedelta_level','user_nexthour_show_level','user_near_timedelta_level','user_nearhour_show_delta_level',
        'cate_age_delta_level',
        'item_his_trade_level','item_lastdate_trade_level','item_lastdate_trade_ratio_level','item_lastdate_trade_delta_level','item_lastdate_trade_ratio_delta_level',
        'item_sales_delta_level','item_collected_delta_level','item_pv_delta_level','item_age_delta_level',
        'item_prop_num_level',
        'shop_lastdate_trade_level','shop_lastdate_trade_ratio_level','shop_lastdate_show_delta_level',
        'shop_age_delta_level',
        'shop_item_count_level',
        'shop_sales_delta_level','shop_collected_delta_level','shop_pv_delta_level',#'shop_review_num_delta_level','shop_star_level_delta_level',
        'brand_his_trade_ratio_level','brand_lastdate_trade_ratio_level',
        'brand_collected_delta_level','brand_price_delta_level','brand_pv_delta_level','brand_sales_delta_level',
        'brand_age_delta_level',
        'shop_brand_item_ratio_level','shop_brand_count_ratio_level',#'shop_brand_special_degree_level',

        # 'uc_next_show_timedelta_level','uc_near_timedelta_level','uc_nearhour_show_delta_level',#'uc_his_trade_ratio_level',
        'uc_his_trade_ratio_level','uc_next_show_timedelta_level','uc_near_timedelta_level','uc_nearhour_show_delta_level',
        'uc_price_mean_level','uc_price_delta_level',
        'ui_last_show_timedelta_level','ui_lasthour_show_level','ui_lasthour_show_ratio_level','ui_lastdate_show_level',
        'ca_his_trade_ratio','cg_his_trade_ratio',
        'ia_his_trade_ratio_level','ia_his_trade_delta_level','ig_his_trade_ratio_level','ig_his_trade_delta_level',
    ]
    fea = numFea + strFea
    print(df[fea].info())
    print(predictDf[fea].info())
    # exit()

    # 正式模型
    modelName = "xgb_lr2B"
    startTime = datetime.now()
    dv = DictDataset(originDf[fea], numFea, strFea)
    dfX = dv.transform(df[fea])
    predictX = dv.transform(predictDf[fea])
    dfy = df['is_trade'].values
    print('transform time: ', datetime.now()-startTime)
    startTime = datetime.now()
    clf = SGDModel()
    # clf.gridSearch(dfX, dfy)
    clf.train(dfX, dfy, save=True, modelName=modelName, verbose=1)
    print('training time: ', datetime.now()-startTime)

    # 开始预测
    predictDf.loc[:,'predicted_score'] = clf.predict(predictX)
    print("预测结果：\n",predictDf[['instance_id','predicted_score']].head())
    print('预测均值：', predictDf['predicted_score'].mean())
    # exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
    # exportResult(predictDf[['instance_id','predicted_score','hour']], "%s_hashour.txt" % modelName)

    # 生成stacking数据集
    df['predicted_score'] = np.nan
    predictDf['predicted_score'] = np.nan
    df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = getOof(clf, dfX, dfy, predictX, stratify=True)#, weight=df['weight'].values
    print('oof training time: ', datetime.now()-startTime)
    cost = metrics.log_loss(df['is_trade'].values, df['predicted_score'].values)
    print('train loss: ', cost)
    print('7th train loss', metrics.log_loss(df.loc[df.date=='2018-09-07','is_trade'].values, df.loc[df.date=='2018-09-07','predicted_score'].values))
    print('train predict: \n',df[['instance_id','predicted_score']].head())
    print('normal dates predict aver:', df.loc[df.context_timestamp<'2018-09-05','predicted_score'].mean())
    print('7th train predict aver:', df.loc[df.date=='2018-09-07','predicted_score'].mean())
    print('test predict: \n',predictDf[['instance_id','predicted_score']].head())
    print('test predict aver:', predictDf['predicted_score'].mean())
    exit()
    exportResult(df[['instance_id','predicted_score']], "%s_oof_train.csv" % modelName)
    exportResult(predictDf[['instance_id','predicted_score']], "%s_oof_test.csv" % modelName)
