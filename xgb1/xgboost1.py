#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： xgboostCV
模型参数：'objective': 'binary:logistic',
        'eval_metric':'logloss',
        'silent': True,
        'eta': 0.05,
        'max_depth': 4,
        'gamma': 0.5,
        'subsample':0.95,
        'colsample_bytree': 0.9,
        'min_child_weight': 8,
        'max_delta_step': 5,
        'lambda': 100,
        'num_boost_round': 1500
        'early_stopping_rounds': 10
        'nfold': 3
特征： 商品销量等级，商品收藏量等级，商品价格等级
      用户性别编号/用户年龄段/用户星级等级/用户职业编号
      小时数，展示页码编号
      店铺好评率/店铺服务评分/店铺物流评分/店铺描述评分

      用户历史日点击数（当天以前），用户历史转化率（当天以前），用户距离上次点击时长（秒）
      商品历史日点击数（当天以前），商品历史转化率（当天以前），商品历史点击数在同类中的占比，商品属性个数
      商品销售等级与同类之差，商品收藏等级与同类之差，商品价格等级与同类之差，商品广告投放等级与同类之差
      店铺历史日点击数，店铺历史交易率
      用户年龄与店铺平均年龄之差
      店铺收藏等级总和/均值，店铺价格总和/均值，店铺广告投放等级总和/均值，店铺销售等级总和/均值，店铺商品数目
      店铺平均销售等级与同类之差，店铺平均收藏等级与同类之差，店铺平均价格与同类之差，店铺平均广告投放与同类之差
      店铺好评率与同行平均差值/店铺服务评分与同行平均差值/店铺物流评分与同行平均差值/店铺描述评分与同行平均差值，店铺评论数与同类平均差值，店铺星级与同类平均差值，店铺商品数目与同类平均差值
      类别历史日点击数，类别历史转化率，
      用户年龄与类别平均年龄差值
      品牌收藏等级总和/均值，类别销售等级总和/均值，类别商品数目
      品牌平均销量与同类之差，品牌平均收藏与同类之差，品牌平均价格与同类之差，品牌平均广告投放与同类之差
      上下文预测类目个数，上下文预测类目是否包含该商品类目，商品包含属性占上下文预测属性的比例
      用户距离上次浏览该商品的时长，用户过去一小时浏览该商品的次数占同类商品的比重
      用户距离上次浏览该类别的时长，用户过去一小时浏览该类别的比重，用户在该类别的历史转化率，用户浏览该类别次数占用户历史浏览的比重
      用户浏览该类别的价格均值，该商品价位与用户浏览该类别的价格均值之差
      用户在该价位的历史转化率，用户在该价位的浏览比例
      商品在该年龄段的浏览比例，商品在该年龄段的转化率，商品在该年龄段的浏览比例与类别在该年龄段的浏览比例之差
      商品在该性别的浏览比例，商品在该性别的转化率，商品在该性别的浏览比例与类别在该性别的浏览比例之差
结果： A榜（0.08107）

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
import xgboost as xgb
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep=' ', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, header=header, index_col=index_col, names=colNames)
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

# 计算单特征与标签的F值
def getFeaScore(X, y, feaNames):
    resultDf = pd.DataFrame(index=feaNames)
    selecter = SelectKBest(f_classif, 'all').fit(X, y)
    resultDf['scores'] = selecter.scores_
    resultDf['p_values'] = selecter.pvalues_
    return resultDf

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

# 转化数据集字段格式，并去重
def formatDf(df):
    df = df.applymap(lambda x: np.nan if (x==-1)|(x=='-1') else x)
    df['context_timestamp'] = df['context_timestamp'].map(lambda x: datetime.fromtimestamp(x))
    return df

# 拆分多维度拼接的字段
def splitMultiFea(df):
    tempDf = df.drop_duplicates(subset=['item_id'])[['item_id','item_category_list','item_property_list']]
    tempDf['item_category_list'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x.split(';'))
    tempDf['item_category0'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[0])
    tempDf['item_category1'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[1] if len(x)>1 else np.nan)
    tempDf['item_category2'] = tempDf[tempDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[2] if len(x)>2 else np.nan)
    tempDf.loc[tempDf.item_category2.notnull(), 'item_category1'] = tempDf.loc[tempDf.item_category2.notnull(), 'item_category2']
    tempDf['item_property_list'] = tempDf[tempDf.item_property_list.notnull()]['item_property_list'].map(lambda x: x.split(';'))
    df = df.drop(['item_category_list','item_property_list'], axis=1).merge(tempDf, how='left', on='item_id')
    df['predict_category_property'] = df[df.predict_category_property.notnull()]['predict_category_property'].map(
        lambda x: {kv.split(':')[0]:(kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) for kv in x.split(';')})
    return df

# 新增多维度拼接的id特征
def combineKey(df):
    df['user_item'] = df['user_id'].astype('str') + '_' + df['item_id'].astype('str')
    df['user_shop'] = df['user_id'].astype('str') + '_' + df['shop_id'].astype('str')
    return df

# 按日期统计过去几天的点击数，交易数
def statDateTrade(df, index, statDates=None, skipDates=1):
    tempDf = pd.pivot_table(df, index=index, columns='date', values='is_trade', aggfunc=[len,np.sum])
    if statDates==None:
        for i,dt in enumerate(tempDf.columns.levels[-1][skipDates:]):
            tempDf.loc[:,pd.IndexSlice['show',dt]] = tempDf['len'].iloc[:,:i+skipDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['trade',dt]] = tempDf['sum'].iloc[:,:i+skipDates].sum(axis=1)
    else:
        for i,dt in enumerate(tempDf.columns.levels[-1][statDates:]):
            tempDf.loc[:,pd.IndexSlice['show',dt]] = tempDf['len'].iloc[:,i:i+statDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['trade',dt]] = tempDf['sum'].iloc[:,i:i+statDates].sum(axis=1)
    tempDf = tempDf.stack()
    tempDf['trade_ratio'] = biasSmooth(tempDf['trade'].values, tempDf['show'].values)
    return tempDf[['show','trade','trade_ratio']]

# 按日期统计过去几天的数目，总和
def statDateLenSum(df, index, values, statDates=None, skipDates=1):
    tempDf = pd.pivot_table(df, index=index, columns='date', values=values, aggfunc=[len,np.sum])
    if statDates==None:
        for i,dt in enumerate(tempDf.columns.levels[-1][skipDates:]):
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,:i+skipDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,:i+skipDates].sum(axis=1)
    else:
        for i,dt in enumerate(tempDf.columns.levels[-1][statDates:]):
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,i:i+statDates].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,i:i+statDates].sum(axis=1)
    tempDf = tempDf.stack()
    return tempDf[['addup_len','addup_sum']]

# 添加时间特征
def addTimeFea(df, **params):
    df['hour'] = df.context_timestamp.dt.hour
    df['hour2'] = ((df['hour']+1) // 2) % 12
    # df['hour2'] = df['hour'] // 2
    df['day'] = df.context_timestamp.dt.day
    df['date'] = pd.to_datetime(df.context_timestamp.dt.date)
    tempDf = statDateTrade(df, 'hour2', **params['statDateTrade'])
    tempDf.columns = ['hour_his_show','hour_his_trade','hour_his_trade_ratio']
    df = df.merge(tempDf, how='left', left_on=['hour2','date'], right_index=True)
    # for x in set(df.hour2.dropna().values):
    #     idx = df[df.hour2==x].index
    #     df.loc[idx, 'hour_his_trade_ratio'] = biasSmooth(df.loc[idx, 'hour_his_trade'].values, df.loc[idx,'hour_his_show'].values)
    return df

# 按维度统计用户年龄特征
def statUserAge(df, originDf, index, prefix):
    tempDf = pd.pivot_table(originDf, index=index, values=['user_age_level'], aggfunc=[np.mean, np.std])
    tempDf.columns = [colname%prefix for colname in ['%s_age_mean','%s_age_std']]
    df = df.merge(tempDf, how='left', left_on=index, right_index=True)
    df['%s_age_delta'%prefix] = df['user_age_level'] - df['%s_age_mean'%prefix]
    return df

# 按维度统计用户性别特征
def statUserGender(df, originDf, index, prefix):
    tempDf = pd.pivot_table(originDf, index=index, values='is_trade', columns=['user_gender_id'], aggfunc=[len])
    startTime = datetime.now()
    sumCol = tempDf.sum(axis=1)
    aver = tempDf['len'].sum() / sumCol.sum()
    for x in tempDf.columns.levels[-1]:
        tempDf.loc[:,pd.IndexSlice['ratio',x]] = biasSmooth(tempDf['len'][x].values, sumCol.values)
        tempDf.loc[:,pd.IndexSlice['delta',x]] = tempDf.loc[:,pd.IndexSlice['ratio',x]] - aver.loc[x]
    del tempDf['len']
    tempDf = tempDf.stack()
    cols = ['%s_gender_%s'%(prefix,x) for x in ['ratio_delta','ratio']]
    tempDf.columns = cols
    df = df.merge(tempDf, how='left', left_on=[index,'user_gender_id'], right_index=True)
    return df

# 统计商品特征
def statItemFea(df, originDf, index, prefix):
    itemDf = originDf.drop_duplicates(['item_id'])
    tempDf = pd.pivot_table(itemDf, index=index, values=['item_sales_level','item_collected_level','item_pv_level','item_price_level'], aggfunc={'item_sales_level':[len,np.sum,np.mean], 'item_collected_level':[np.sum,np.mean], 'item_pv_level':[np.sum,np.mean], 'item_price_level':[np.sum,np.mean]})
    tempDf.columns = [col%prefix for col in ['%s_collected_mean','%s_collected_sum','%s_price_mean','%s_price_sum','%s_pv_mean','%s_pv_sum','%s_item_count','%s_sales_mean','%s_sales_sum']]
    df = df.merge(tempDf, how='left', left_on=index,right_index=True)
    return df

# 添加商品类别统计特征
def addCateFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, 'item_category1', **params['statDateTrade'])
    tempDf.columns = ['cate_his_show','cate_his_trade','cate_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['cate_his_show_perday'] = tempDf['cate_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['item_category1','date'], right_index=True)

    df = statItemFea(df, originDf, 'item_category1', 'cate')
    df = statUserAge(df, originDf, 'item_category1', 'cate')
    df = statUserGender(df, originDf, 'item_category1', 'cate')
    df = statUserStar(df, originDf, 'item_category1', 'cate')
    # df = statUserJob(df, originDf, 'item_category1', 'cate')
    return df

# 添加商品历史浏览量及购买量特征
def addShopFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(df, 'shop_id', **params['statDateTrade'])
    tempDf.columns = ['shop_his_show','shop_his_trade','shop_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['shop_his_show_perday'] = tempDf['shop_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['shop_id','date'], right_index=True)
    df['shop_his_show_ratio'] = biasSmooth(df.shop_his_show.values, df.cate_his_show.values)

    df = statItemFea(df, df, 'shop_id', 'shop')
    df['shop_sales_delta'] = df['shop_sales_mean'] - df['cate_sales_mean']
    df['shop_collected_delta'] = df['shop_collected_mean'] - df['cate_collected_mean']
    df['shop_price_delta'] = df['shop_price_mean'] - df['cate_price_mean']
    df['shop_pv_delta'] = df['shop_pv_mean'] - df['cate_pv_mean']
    tempDf = pd.pivot_table(df.drop_duplicates(['shop_id','date']), index=['item_category1','date'], values=['shop_his_show_perday'], aggfunc=np.mean)
    tempDf.columns = ['shop_his_show_mean']
    df = df.merge(tempDf, how='left', left_on=['item_category1','date'], right_index=True)
    df['shop_his_show_delta'] = df['shop_his_show_perday'] - df['shop_his_show_mean']

    shopDf = df.drop_duplicates(['item_category1','shop_id'], keep='last')
    tempDf = pd.pivot_table(shopDf, index=['item_category1'], values=['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description','shop_star_level','shop_review_num_level','shop_item_count'], aggfunc=np.mean)
    tempDf.columns = ['shop_item_count_mean','shop_review_num_mean','shop_review_positive_mean','shop_score_delivery_mean','shop_score_description_mean','shop_score_service_mean','shop_star_level_mean']
    tempDf.reset_index(inplace=True)
    df = df.merge(tempDf, how='left', on='item_category1')
    df['shop_review_num_delta'] = df['shop_review_num_level'] - df['shop_review_num_mean']
    df['shop_review_positive_delta'] = df['shop_review_positive_rate'] - df['shop_review_positive_mean']
    df['shop_score_service_delta'] = df['shop_score_service'] - df['shop_score_service_mean']
    df['shop_score_delivery_delta'] = df['shop_score_delivery'] - df['shop_score_delivery_mean']
    df['shop_score_description_delta'] = df['shop_score_description'] - df['shop_score_description_mean']
    df['shop_star_level_delta'] = df['shop_star_level'] - df['shop_star_level_mean']
    df['shop_item_count_delta'] = df['shop_item_count'] - df['shop_item_count_mean']

    df = statUserAge(df, df, 'shop_id', 'shop')
    df = statUserGender(df, df, 'shop_id', 'shop')
    # df = statUserStar(df, df, 'shop_id', 'shop')
    # df = statUserJob(df, df, 'shop_id', 'shop')
    return df

# 添加商品历史浏览量及购买量特征
def addItemFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, 'item_id', **params['statDateTrade'])
    tempDf.columns = ['item_his_show','item_his_trade','item_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['item_his_show_perday'] = tempDf['item_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['item_id','date'], right_index=True)
    df['item_his_show_ratio'] = biasSmooth(df.item_his_show.values, df.cate_his_show.values)
    df['item_prop_num'] = df['item_property_list'].dropna().map(lambda x: len(x))

    df['item_sales_delta'] = df['item_sales_level'] - df['cate_sales_mean']
    df['item_collected_delta'] = df['item_collected_level'] - df['cate_collected_mean']
    df['item_price_delta'] = df['item_price_level'] - df['cate_price_mean']
    df['item_pv_delta'] = df['item_pv_level'] - df['cate_pv_mean']
    tempDf = pd.pivot_table(df.drop_duplicates(['item_id','date']), index=['item_category1','date'], values=['item_his_show_perday'], aggfunc=np.mean)
    tempDf.columns = ['item_his_show_mean']
    df = df.merge(tempDf, how='left', left_on=['item_category1','date'], right_index=True)
    df['item_his_show_delta'] = df['item_his_show_perday'] - df['item_his_show_mean']

    df = statUserAge(df, originDf, 'item_id', 'item')
    df = statUserGender(df, originDf, 'item_id', 'item')
    # df = statUserStar(df, originDf, 'item_id', 'item')
    # df = statUserJob(df, originDf, 'item_id', 'item')
    return df

# 添加用户维度特征
def addUserFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, 'user_id', **params['statDateTrade'])
    tempDf.columns = ['user_his_show','user_his_trade','user_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['user_his_show_perday'] = tempDf['user_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['user_id','date'], right_index=True)

    tempDf = pd.pivot_table(originDf, index=['user_id','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_id'] = tempDf['user_id'].shift(1)
    tempDf['last_user_id'] = tempDf['last_user_id']==tempDf['user_id']
    tempDf['last_show_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_user_id, 'last_show_time'] = np.nan
    tempDf['user_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_show_time']
    tempDf['user_last_show_timedelta'] = tempDf['user_last_show_timedelta'].dt.seconds
    tempDf['user_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_id','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['user_lasthour_show'] = hourShowList
    df = df.merge(tempDf[['user_id','context_timestamp','user_last_show_timedelta','user_lasthour_show']], how='left', on=['user_id','context_timestamp'])
    return df

# 添加广告商品与查询词的相关性特征
def addContextFea(df, **params):
    df['predict_category'] = df['predict_category_property'].dropna().map(lambda x: list(x.keys()))
    df['predict_cate_num'] = df['predict_category'].dropna().map(lambda x: len(x))
    df.loc[df.predict_category_property.notnull(),'has_predict_category'] = list(map(lambda x: 1 if x[0] in x[1] else 0, df.loc[df.predict_category.notnull(), ['item_category1','predict_category']].values))
    df.loc[df.has_predict_category==1,'predict_property'] = list(map(lambda x: x[1][x[0]], df.loc[df.has_predict_category==1, ['item_category1','predict_category_property']].values))
    df['predict_prop_num'] = df[df.predict_property.notnull()]['predict_property'].map(lambda x: len(x))
    df.loc[df.predict_property.notnull(), 'has_predict_prop_num'] = list(map(lambda x: len(np.intersect1d(x[0],x[1])), df.loc[df.predict_property.notnull(), ['item_property_list','predict_property']].values))
    df.loc[df.predict_prop_num>0,'has_predict_prop_ratio'] = df['has_predict_prop_num'] / df['predict_prop_num']
    df.fillna({k:0 for k in ['predict_cate_num','predict_prop_num','has_predict_prop_num','has_predict_category']}, inplace=True)
    return df

# 添加品牌相关特征
def addBrandFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    brandDf = df.drop_duplicates(['item_id']).item_brand_id.value_counts()
    dropBrand = brandDf[brandDf<5].index
    df.loc[~df.item_brand_id.isin(dropBrand),'item_brand_id'] = np.nan
    tempDf = statDateTrade(originDf, 'item_brand_id', **params['statDateTrade'])
    tempDf.columns = ['brand_his_show','brand_his_trade','brand_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['brand_his_show_perday'] = tempDf['brand_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','date'], right_index=True)

    df = statItemFea(df, originDf, 'item_brand_id', 'brand')
    df['brand_sales_delta'] = df['brand_sales_mean'] - df['cate_sales_mean']
    df['brand_collected_delta'] = df['brand_collected_mean'] - df['cate_collected_mean']
    df['brand_price_delta'] = df['brand_price_mean'] - df['cate_price_mean']
    df['brand_pv_delta'] = df['brand_pv_mean'] - df['cate_pv_mean']

    df = statUserAge(df, originDf, 'item_brand_id', 'brand')
    df = statUserGender(df, originDf, 'item_brand_id', 'brand')
    # df = statUserStar(df, originDf, 'item_brand_id', 'brand')
    # df = statUserJob(df, originDf, 'item_brand_id', 'brand')
    return df

# 添加用户与类目关联维度的特征
def addUserCateFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = pd.pivot_table(originDf, index=['user_id','item_category1','context_timestamp'], values=['item_price_level'], aggfunc=[len,np.mean])
    tempDf.columns = ['show','price']
    tempDf.reset_index(inplace=True)
    tempDf[['last_user','last_cate','last_time']] = tempDf[['user_id','item_category1','context_timestamp']].shift(1)
    tempDf['same'] = (tempDf.user_id==tempDf.last_user) & (tempDf.item_category1==tempDf.last_cate)
    tempDf.loc[~tempDf.same, 'last_time'] = np.nan
    tempDf['uc_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_time']
    tempDf['uc_last_show_timedelta'] = tempDf['uc_last_show_timedelta'].dt.seconds
    tempDf['uc_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    priceList = []
    tempPrice = []
    for same, dt, show, price in tempDf[['same','context_timestamp','show','price']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
            priceList.append(tempPrice)
            tempPrice = tempPrice+[price]
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
            priceList.append(np.nan)
            tempPrice = [price]
    tempDf['uc_lasthour_show'] = hourShowList
    tempDf['uc_price_mean'] = priceList
    tempDf['uc_price_mean'] = tempDf.loc[tempDf.uc_price_mean.notnull(),'uc_price_mean'].map(lambda x: np.mean(x))
    df = df.merge(tempDf[['user_id','item_category1','context_timestamp','uc_last_show_timedelta','uc_lasthour_show','uc_price_mean']], how='left', on=['user_id','item_category1','context_timestamp'])
    df['uc_lasthour_show_ratio'] = biasSmooth(df.uc_lasthour_show.values, df.user_lasthour_show.values)
    df['uc_price_delta'] = df['item_price_level'] - df['uc_price_mean']

    tempDf = statDateTrade(originDf, ['user_id','item_category1'], **params['statDateTrade'])
    tempDf.columns = ['uc_his_show','uc_his_trade','uc_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['uc_his_show_perday'] = tempDf['uc_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['user_id','item_category1','date'], right_index=True)
    # df['uc_his_show_ratio'] = biasSmooth(df['uc_his_show'].values, df.user_his_show.values)
    for x in set(df.item_category1.dropna().values):
        idx = df[df.item_category1==x].index
        df.loc[idx, 'uc_his_show_ratio'] = biasSmooth(df.loc[idx, 'uc_his_show'].values, df.loc[idx,'user_his_show'].values)
    return df

# 添加用户商品关联维度的特征
def addUserItemFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = pd.pivot_table(originDf, index=['user_item','context_timestamp'], values=['is_trade'], aggfunc=len)
    tempDf.columns = ['show']
    tempDf.reset_index(inplace=True)
    tempDf['last_user_item'] = tempDf['user_item'].shift(1)
    tempDf['last_user_item'] = tempDf['last_user_item']==tempDf['user_item']
    tempDf['last_show_time'] = tempDf['context_timestamp'].shift(1)
    tempDf.loc[~tempDf.last_user_item, 'last_show_time'] = np.nan
    tempDf['ui_last_show_timedelta'] = tempDf['context_timestamp'] - tempDf['last_show_time']
    tempDf['ui_last_show_timedelta'] = tempDf['ui_last_show_timedelta'].dt.seconds
    tempDf['ui_last_show_timedelta'].fillna(999999, inplace=True)
    hourShowList = []
    hourShowTemp = {}
    for same, dt, show in tempDf[['last_user_item','context_timestamp','show']].values:
        if same:
            [hourShowTemp.pop(k) for k in list(hourShowTemp) if k<dt-timedelta(hours=1)]
            hourShowList.append(np.sum(list(hourShowTemp.values())))
            hourShowTemp[dt] = show
        else:
            hourShowList.append(0)
            hourShowTemp = {dt:show}
    tempDf['ui_lasthour_show'] = hourShowList
    df = df.merge(tempDf[['user_item','context_timestamp','ui_last_show_timedelta','ui_lasthour_show']], how='left', on=['user_item','context_timestamp'])
    df['ui_lasthour_show_ratio'] = biasSmooth(df.ui_lasthour_show.values, df.uc_lasthour_show.values)

    tempDf = pd.pivot_table(originDf, index=['user_item','date'], values=['is_trade'], aggfunc=[len,np.sum])
    tempDf.columns = ['ui_lastdate_show', 'ui_lastdate_trade']
    tempDf.reset_index(inplace=True)
    tempDf['date'] = tempDf['date'] + timedelta(days=1)
    df = df.merge(tempDf[['user_item','date','ui_lastdate_show', 'ui_lastdate_trade']], how='left', on=['user_item','date'])
    df.fillna({k:0 for k in ['ui_lastdate_show', 'ui_lastdate_trade']}, inplace=True)
    return df

# 统计用户该价格段商品的统计特征
def addUserShopFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','shop_id'], **params['statDateTrade'])
    tempDf.columns = ['us_his_show','us_his_trade','us_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['us_his_show_perday'] = tempDf['us_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['user_id','shop_id','date'], right_index=True)
    return df

# 统计用户该价格段商品的统计特征
def addUserPriceFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','item_price_level'], **params['statDateTrade'])
    tempDf.columns = ['up_his_show','up_his_trade','up_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['up_his_show_perday'] = tempDf['up_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['user_id','item_price_level','date'], right_index=True)
    # df['up_his_show_ratio'] = biasSmooth(df.up_his_show.values, df.user_his_show.values)
    for x in set(df.item_price_level.dropna().values):
        idx = df[df.item_price_level==x].index
        df.loc[idx, 'up_his_show_ratio'] = biasSmooth(df.loc[idx, 'up_his_show'].values, df.loc[idx,'user_his_show'].values)
    return df

# 统计用户该价格段商品的统计特征
def addUserHourFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    tempDf = statDateTrade(originDf, ['user_id','hour2'], **params['statDateTrade'])
    tempDf.columns = ['uh_his_show','uh_his_trade','uh_his_trade_ratio']
    startDate = pd.to_datetime('2018-09-17')
    startDate += timedelta(days=params['statDateTrade']['skipDates']) if params['statDateTrade']['statDates']==None else timedelta(days=params['statDateTrade']['statDates'])
    tempDf['uh_his_show_perday'] = tempDf['uh_his_show'] / (tempDf.index.get_level_values('date')-startDate).days
    df = df.merge(tempDf, how='left', left_on=['user_id','hour2','date'], right_index=True)
    # df['uh_his_show_ratio'] = biasSmooth(df.uh_his_show.values, df.user_his_show.values)
    for x in set(df.hour2.dropna().values):
        idx = df[df.hour2==x].index
        df.loc[idx, 'uh_his_show_ratio'] = biasSmooth(df.loc[idx, 'uh_his_show'].values, df.loc[idx,'user_his_show'].values)
    return df

# 统计交叉转化率特征
def addCrossTradeFea(df, index, prefix, params):
    tempDf = statDateTrade(df, index, **params['statDateTrade'])
    tempDf.columns = ['%s_his_show'%prefix,'%s_his_trade'%prefix,'%s_his_trade_ratio'%prefix]
    df = df.merge(tempDf, how='left', left_on=index+['date'], right_index=True)
    return df

# 统计商品与各类别用户群交叉特征
def addCateCrossFea(df, **params):
    # df = addCrossTradeFea(df, ['item_category1','user_age_level'], 'ca', params)
    tempDf = statDateTrade(df, ['item_category1','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ca_his_show','ca_his_trade','ca_his_trade_ratio']
    df = df.merge(tempDf, how='left', left_on=['item_category1','user_age_level','date'], right_index=True)
    df['ca_his_show_ratio'] = biasSmooth(df.ca_his_show.values, df.cate_his_show.values)
    # df = addCrossTradeFea(df, ['item_category1','user_gender_id'], 'cg', params)
    tempDf = statDateTrade(df, ['item_category1','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['cg_his_show','cg_his_trade','cg_his_trade_ratio']
    df = df.merge(tempDf, how='left', left_on=['item_category1','user_gender_id','date'], right_index=True)
    df['cg_his_show_ratio'] = biasSmooth(df.cg_his_show.values, df.cate_his_show.values)
    return df

# 统计商品与各类别用户群交叉特征
def addItemCrossFea(df, **params):
    # df = addCrossTradeFea(df, ['item_id','user_age_level'], 'ia', params)
    tempDf = statDateTrade(df, ['item_id','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ia_his_show','ia_his_trade','ia_his_trade_ratio']
    df = df.merge(tempDf, how='left', left_on=['item_id','user_age_level','date'], right_index=True)
    for x in set(df.user_age_level.dropna().values):
        idx = df[df.user_age_level==x].index
        df.loc[idx, 'ia_his_show_ratio'] = biasSmooth(df.loc[idx, 'ia_his_show'].values, df.loc[idx,'item_his_show'].values)
    df['ia_his_show_delta'] = df['ia_his_show_ratio'] - df['ca_his_show_ratio']
    # df = addCrossTradeFea(df, ['item_id','user_gender_id'], 'ig', params)
    tempDf = statDateTrade(df, ['item_id','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['ig_his_show','ig_his_trade','ig_his_trade_ratio']
    df = df.merge(tempDf, how='left', left_on=['item_id','user_gender_id','date'], right_index=True)
    for x in set(df.user_gender_id.dropna().values):
        idx = df[df.user_gender_id==x].index
        df.loc[idx, 'ig_his_show_ratio'] = biasSmooth(df.loc[idx, 'ig_his_show'].values, df.loc[idx,'item_his_show'].values)
    df['ig_his_show_delta'] = df['ig_his_show_ratio'] - df['cg_his_show_ratio']
    return df

# 统计品牌与各类别用户群交叉特征
def addBrandCrossFea(df, **params):
    # df = addCrossTradeFea(df, ['item_brand_id','user_age_level'], 'ia', params)
    tempDf = statDateTrade(df, ['item_brand_id','user_age_level'], **params['statDateTrade'])
    tempDf.columns = ['ba_his_show','ba_his_trade','ba_his_trade_ratio']
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_age_level','date'], right_index=True)
    for x in set(df.user_age_level.dropna().values):
        idx = df[df.user_age_level==x].index
        df.loc[idx, 'ba_his_show_ratio'] = biasSmooth(df.loc[idx, 'ba_his_show'].values, df.loc[idx,'item_his_show'].values)
    df['ba_his_show_delta'] = df['ba_his_show_ratio'] - df['ca_his_show_ratio']
    # df = addCrossTradeFea(df, ['item_brand_id','user_gender_id'], 'ig', params)
    tempDf = statDateTrade(df, ['item_brand_id','user_gender_id'], **params['statDateTrade'])
    tempDf.columns = ['bg_his_show','bg_his_trade','bg_his_trade_ratio']
    df = df.merge(tempDf, how='left', left_on=['item_brand_id','user_gender_id','date'], right_index=True)
    for x in set(df.user_age_level.dropna().values):
        idx = df[df.user_age_level==x].index
        df.loc[idx, 'bg_his_show_ratio'] = biasSmooth(df.loc[idx, 'bg_his_show'].values, df.loc[idx,'item_his_show'].values)
    df['bg_his_show_ratio'] = biasSmooth(df.bg_his_show.values, df.item_his_show.values)
    df['bg_his_show_delta'] = df['bg_his_show_ratio'] - df['cg_his_show_ratio']
    return df

def addItemCateFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    df['ic_price_delta'] = df['item_price_level'] - df['cate_price_mean']
    df['ic_sales_delta'] = df['item_sales_level'] - df['cate_sales_mean']
    # df['ic_trade_ratio_delta'] = df['item_his_trade_ratio'] - df['cate_his_trade_ratio']
    return df

def addItemShopFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    df['is_sales_delta'] = df['item_sales_level'] - df['shop_sales_mean']
    df['is_collected_delta'] = df['item_collected_level'] - df['shop_collected_mean']
    df['is_pv_delta'] = df['item_pv_level'] - df['shop_pv_mean']
    return df

def addBrandCateFea(df, originDf=None, **params):
    if not isinstance(originDf, pd.DataFrame):
        originDf = df.copy()
        originDf['is_trade'].fillna(0,inplace=True)
    df['bc_sales_delta'] = df['brand_sales_mean'] - df['cate_sales_mean']
    df['bc_collected_delta'] = df['brand_collected_mean'] - df['cate_collected_mean']
    return df

class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric':'logloss',
            'silent': True,
            'eta': 0.05,
            'max_depth': 4,
            'gamma': 0.5,
            'subsample': 0.95,
            'colsample_bytree': 1,
            'min_child_weight': 8,
            'max_delta_step': 5,
            'lambda': 100,
        }
        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
        X = X.astype(float)
        if train_size==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feaNames)
        dval = xgb.DMatrix(X_test, label=y_test, feature_names=self.feaNames)
        watchlist = [(dtrain,'train'),(dval,'val')]
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            evals = watchlist, 
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        self.clf = clf

    def trainCV(self, X, y, nFold=3, verbose=True, num_boost_round=1500, early_stopping_rounds=10):
        X = X.astype(float)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        cvResult = xgb.cv(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            nfold = nFold,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = cvResult.shape[0], 
        )
        self.clf = clf

    def gridSearch(self, X, y, nFold=3, verbose=1, num_boost_round=130):
        paramsGrids = {
            # 'n_estimators': [50+5*i for i in range(0,30)],
            'gamma': [0,0.01,0.05,0.1,0.5,1,5,10,50,100],
            # 'max_depth': list(range(3,10)),
            'min_child_weight': list(range(0,10)),
            'subsample': [1-0.05*i for i in range(0,8)],
            'colsample_bytree': [1-0.05*i for i in range(0,10)],
            # 'reg_alpha': [0+2*i for i in range(0,10)],
            'reg_lambda': [0+50*i for i in range(0,10)],            
            'max_delta_step': [0+1*i for i in range(0,8)],
        }
        for k,v in paramsGrids.items():
            gsearch = GridSearchCV(
                estimator = xgb.XGBClassifier(
                    max_depth = self.params['max_depth'], 
                    gamma = self.params['gamma'],
                    learning_rate = self.params['eta'],
                    max_delta_step = self.params['max_delta_step'],
                    min_child_weight = self.params['min_child_weight'],
                    subsample = self.params['subsample'],
                    colsample_bytree = self.params['colsample_bytree'],
                    silent = self.params['silent'],
                    reg_lambda = self.params['lambda'],
                    n_estimators = num_boost_round
                ),
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
        exit()

    def predict(self, X):
        X = X.astype(float)
        return self.clf.predict(xgb.DMatrix(X, feature_names=self.feaNames))

    def getFeaScore(self, show=False):
        fscore = self.clf.get_score()
        feaNames = fscore.keys()
        scoreDf = pd.DataFrame(index=feaNames, columns=['importance'])
        for k,v in fscore.items():
            scoreDf.loc[k, 'importance'] = v
        if show:
            print(scoreDf.sort_index(by=['importance'], ascending=False))
        return scoreDf

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
    df = formatDf(df)
    df = splitMultiFea(df)
    df = combineKey(df)
    df = addTimeFea(df, **params)
    df = addCateFea(df, originDf, **params)
    df = addUserFea(df, originDf, **params)
    df = addShopFea(df, originDf, **params)
    df = addItemFea(df, originDf, **params)
    df = addContextFea(df, **params)
    df = addBrandFea(df,originDf, **params)
    df = addUserCateFea(df, originDf, **params)
    df = addUserItemFea(df, originDf, **params)
    df = addUserShopFea(df, originDf, **params)
    df = addUserPriceFea(df, originDf, **params)
    df = addUserHourFea(df, originDf, **params)
    df = addCateCrossFea(df, **params)
    df = addBrandCrossFea(df, **params)
    df = addItemCrossFea(df, **params)

    # df = addItemCateFea(df, originDf, **params)
    # df = addItemShopFea(df, originDf, **params)
    # df = addBrandCateFea(df, originDf, **params)
    print('finished feaFactory: ', datetime.now() - startTime)
    return df

# 划分训练集和测试集
def trainTestSplit(df, splitDate=pd.to_datetime('2018-09-23'), trainPeriod=3, testPeriod=1):
    trainDf = df[(df.context_timestamp<splitDate)&(df.context_timestamp>=splitDate-timedelta(days=trainPeriod))]
    testDf = df[(df.context_timestamp>=splitDate)&(df.context_timestamp<splitDate+timedelta(days=testPeriod))]
    return (trainDf, testDf)

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=' '):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 统计预测误差
def countDeltaY(predictSeries, labelSeries, show=True, title='', subplot=None):
    deltaSeries = predictSeries - labelSeries
    if subplot!=None:
        plt.subplot(subplot[0], subplot[1], subplot[2])
    deltaSeries.plot(style='b-')
    plt.title(title)
    if show:
        plt.show()
    return deltaSeries

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=5, stratify=False):
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
        clf.trainCV(kfTrainX, kfTrainY, verbose=False)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
    oofTest[:] = oofTestSkf.mean(axis=1)
    return oofTrain, oofTest


if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/round1_ijcai_18_train_20180301.txt')
    predictDf = importDf('../data/round1_ijcai_18_test_a_20180301.txt')

    # 特征处理
    df.drop_duplicates(inplace=True)
    originDf = pd.concat([df,predictDf], ignore_index=True)
    startIdx = len(df)
    originDf = feaFactory(originDf)
    df = originDf.iloc[:startIdx]
    predictDf = originDf.iloc[startIdx:]

    # 特征筛选
    tempCol = ['hour_his_trade_ratio']
    resultDf = getFeaScore(df.dropna(subset=tempCol)[tempCol].values, df.dropna(subset=tempCol)['is_trade'].values, tempCol)
    print(resultDf[resultDf.scores>0])
    print(df[tempCol].describe())
    fea = [
        'item_sales_level','item_price_level','item_collected_level',#'item_category1','item_city_id',
        'user_gender_id','user_age_level','user_star_level','user_occupation_id',
        'hour','context_page_id',#'hour2',
        'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description',#'shop_star_level',

        'user_his_show_perday','user_his_trade_ratio','user_last_show_timedelta',#'user_his_show','user_his_trade','user_lasthour_show',
        'item_his_trade_ratio','item_his_show_ratio','item_prop_num','item_his_show_perday',#'item_his_show','item_his_trade','item_his_show_delta',
        # 'item_age_delta',#'item_age_mean','item_age_std',
        # 'item_gender_ratio_delta',#'item_gender_ratio',
        'item_sales_delta','item_collected_delta','item_price_delta','item_pv_delta',
        'shop_his_show_perday','shop_his_trade_ratio',#'shop_his_show_ratio',#'shop_his_trade','shop_his_show','shop_his_show_delta',
        'shop_age_delta',#'shop_age_std','shop_age_mean',
        # 'shop_gender_ratio_delta',#'shop_gender_ratio',
        'shop_collected_mean','shop_collected_sum','shop_price_mean','shop_price_sum','shop_pv_mean','shop_pv_sum','shop_sales_mean','shop_sales_sum','shop_item_count',
        'shop_sales_delta','shop_collected_delta','shop_price_delta','shop_pv_delta',
        'shop_score_service_delta','shop_score_delivery_delta','shop_score_description_delta','shop_review_positive_delta','shop_review_num_delta','shop_star_level_delta','shop_item_count_delta',
        'cate_his_show_perday','cate_his_trade_ratio',#'cate_his_trade','cate_his_show',
        'cate_age_delta',#'cate_age_std','cate_age_mean',
        # 'cate_gender_ratio_delta',#'cate_gender_ratio',
        # 'cate_price_mean','cate_sales_mean','cate_sales_sum','cate_collected_mean','cate_collected_sum','cate_price_sum','cate_pv_mean','cate_pv_sum','cate_item_count',
        # 'brand_his_show_perday','brand_his_trade_ratio',#'brand_his_trade','brand_his_show',
        'brand_item_count','brand_collected_mean','brand_collected_sum','brand_sales_mean','brand_sales_sum',
        'brand_sales_delta','brand_collected_delta','brand_price_delta','brand_pv_delta',
        # 'brand_age_std','brand_age_delta','brand_age_mean',
        # 'brand_gender_ratio_delta',#'brand_gender_ratio',
        # 'brand_star_delta',#'brand_star_mean','brand_star_std',
        # 'brand_job_ratio_delta',#'brand_job_ratio',
        'predict_cate_num','has_predict_category','has_predict_prop_ratio',#'has_predict_prop_num',
        # 'hour_his_trade_ratio',

        'ui_last_show_timedelta','ui_lasthour_show_ratio',#'ui_lasthour_show',#'ui_lastdate_trade',#'ui_lastdate_show',
        'uc_his_trade_ratio','uc_last_show_timedelta','uc_his_show_ratio','uc_lasthour_show_ratio',# 'uc_his_trade','uc_his_show','uc_lasthour_show',
        'uc_price_mean','uc_price_delta',
        'up_his_trade_ratio','up_his_show_ratio',#'up_his_trade','up_his_show',
        # 'us_his_trade',# 'us_his_trade_ratio','us_his_show',#'us_his_show_perday',
        # 'uh_his_show_ratio',#'uh_his_trade_ratio','uh_his_show',#'uh_his_show_perday',
        'ia_his_show_ratio','ia_his_trade_ratio','ia_his_show_delta',
        'ig_his_show_ratio','ig_his_trade_ratio','ig_his_show_delta',
        # 'ba_his_show_ratio','ba_his_trade_ratio','ba_his_show_delta',
        # 'bg_his_show_ratio','bg_his_trade_ratio','bg_his_show_delta',
        # 'bc_sales_delta','bc_collected_delta',
    ]
    print(df[fea].info())


    # 测试模型效果
    costDf = pd.DataFrame(index=fea+['cost','oof_cost'])
    xgbModel = XgbModel(feaNames=fea)
    for dt in pd.date_range(start='2018-09-23', end='2018-09-24', freq='D'):
        trainDf, testDf = trainTestSplit(df, dt, trainPeriod=4)
        # xgbModel.gridSearch(trainDf[fea].values, trainDf['is_trade'].values)
        xgbModel.trainCV(trainDf[fea].values, trainDf['is_trade'].values, nFold=3)
        testDf.loc[:,'predict'] = xgbModel.predict(testDf[fea].values)
        # _,testDf.loc[:,'oof_predict'] = getOof(xgbModel, trainDf[fea].values, trainDf['is_trade'].values, testDf[fea].values, stratify=True)
        scoreDf = xgbModel.getFeaScore()
        scoreDf.columns = [dt.strftime('%Y-%m-%d')]
        costDf = costDf.merge(scoreDf, how='left', left_index=True, right_index=True)
        cost = metrics.log_loss(testDf['is_trade'].values, testDf['predict'].values)
        costDf.loc['cost',dt.strftime('%Y-%m-%d')] = cost
        # costDf.loc['oof_cost',dt.strftime('%Y-%m-%d')] = metrics.log_loss(testDf['is_trade'].values, testDf['oof_predict'].values)
    print(costDf.iloc[:50], costDf.iloc[50:])
    exit()

    # 正式模型
    modelName = "xgboost1A"
    df = df[df.context_timestamp>='2018-09-19']
    xgbModel.trainCV(df[fea].values, df['is_trade'].values)
    xgbModel.getFeaScore(show=True)
    xgbModel.clf.save_model('%s.model'%modelName)

    # 预测集准备
    startTime = datetime.now()
    print(predictDf[fea].info())

    # 开始预测
    predictDf.loc[:,'predicted_score'] = xgbModel.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['instance_id','predicted_score']].head())
    print('预测均值：', predictDf['predicted_score'].mean())
    exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)

    # 生成stacking数据集
    df['predicted_score'] = np.nan
    predictDf['predicted_score'] = np.nan
    df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = getOof(xgbModel, df[fea].values, df['is_trade'].values, predictDf[fea].values, stratify=True)
    exportResult(df[['instance_id','predicted_score']], "%s_oof_train.csv" % modelName)
    exportResult(predictDf[['instance_id','predicted_score']], "%s_oof_test.csv" % modelName)
