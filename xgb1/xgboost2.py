#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： xgboostCV
模型参数：'objective': 'binary:logistic',
        'eval_metric':'logloss',
        'silent': True,
        'eta': 0.1,
        'max_depth': 4,
        'gamma': 0.5,
        'subsample':0.95,
        'colsample_bytree': 1,
        'min_child_weight': 8,
        'max_delta_step': 5,
        'lambda': 100,
        'num_boost_round': 1500
        'early_stopping_rounds': 10
        'nfold': 3
特征： 商品销量等级，商品收藏量等级，商品价格等级，商品广告等级
      用户性别编号，用户年龄段，用户星级等级，用户职业编号
      展示页码编号
      店铺好评率/店铺服务评分/店铺物流评分/店铺描述评分，店铺星级，店铺评论数

      用户历史转化率（当天以前），用户距离上次点击时长（秒），用户历史点击量，用户过去一小时点击量，用户前一天点击量，用户前一天交易率
      商品历史转化率（当天以前），商品属性个数，商品前一天交易量，商品历史交易量，商品前一天转化率，商品前一天交易量与同类差值，商品前一天转化率与同类差值，商品前一天点击量与同类差值，商品前一天点击量
      用户年龄与商品平均用户年龄差值
      商品销售等级与同类之差，商品收藏与同类之差，商品广告与同类之差
      店铺历史交易率，店铺前一天交易量，店铺前一天转化率，店铺前一天点击量与同类差值
      用户年龄与店铺平均用户年龄差值
      店铺商品数目
      店铺平均销量与同类之差，店铺平均收藏与同类之差，店铺平均广告投放与同类之差
      店铺评论数与同类之差，店铺星级与同类之差，店铺商品数与同类之差
      类目历史转化率，二级类目历史转化率
      用户年龄与类目平均年龄差值
      类目销量均值
      品牌历史转化率，品牌前一天转化率，
      品牌价格与同类差值，品牌收藏与同类差值，品牌销量与同类差值，品牌广告与同类差值
      用户年龄与品牌平均年龄差值
      店铺在该品牌的商品比例，店铺品牌数与商品数比值，店铺品牌专卖程度
      小时转化率，上下文预测类目个数，上下文预测类目与商品类目交集个数，上下文属性与商品属性交集的数目，上下文预测属性个数，

      用户距离上次点击该商品时长，用户过去一小时点击该商品次数，用户前一天点击该商品次数，用户过去一小时点击该商品与点击该类目次数的比例，
      用户在该类目的历史转化率
      用户在该类别的价格均值，商品价格与用户在该类目价格均值的差值
      用户距离上次点击该店铺时长，用户在该店铺过去一小时点击数，用户在该店铺过去一小时点击数与点击该类目次数的比例
      类目在该年龄段的转化率，类目在该性别的转化率
      商品在该年龄段的转化率，商品在该年龄段的转化率与同类之差
      商品在该性别的转化率，商品在该性别的转化率与同类之差
      品牌在该年龄段的转化率，品牌在该年龄转化率与同类之差
      品牌在该性别的转化率，品牌在该性别转化率与同类之差，品牌在该性别点击率与同类之差

      用户距离下次点击时长，用户未来一小时点击量，用户前后两次点击时长的差值，用户前后一小时点击量差值，
      用户在该类目前后一小时点击量差值，用户在该类目前后两次点击时长差值，用户距离下次点击该类目时长，用户在该类目未来一小时点击量
      用户距离下次点击该商品时长，用户未来一小时点击该商品次数，用户前后两次点击该商品的时长差值
      用户距离下次点击该店铺时长，用户前后一小时点击该店铺次数差值，用户在该店铺未来一小时的点击次数
前期处理：采用全部数据集做统计
后期处理：根据普通日期曲线与特殊日期曲线估算出7号下午每小时的转化率，将训练结果各小时的转化率调整至估算的转化率，再将整体均值调到线上均值
结果： 只用7号训练，线下16952
      7号 + 15%普通日期， 线下16946
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


class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        # self.params = {
        #     'objective': 'binary:logistic',
        #     'eval_metric':'logloss',
        #     'silent': True,
        #     'eta': 0.01,
        #     'max_depth': 6,
        #     'gamma': 0.1,
        #     'subsample': 0.886,
        #     'colsample_bytree': 0.886,
        #     'min_child_weight': 1.2,
        #     # 'max_delta_step': 5,
        #     'lambda': 5,
        # }
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric':'logloss',
            'silent': True,
            'eta': 0.1,
            'max_depth': 4,
            'gamma': 0.5,
            'subsample': 0.95,
            'colsample_bytree': 1,
            'min_child_weight': 8,
            'max_delta_step': 5,
            'lambda': 100,
            # 'nthread': 20,
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

    def trainCV(self, X, y, nFold=3, verbose=True, num_boost_round=1500, early_stopping_rounds=10, weight=None):
        X = X.astype(float)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        if weight!=None:
            dtrain.set_weight(weight)
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

# 特征工程
def feaFactory(df):
    df.loc[df.uc_near_timedelta.notnull(),'uc_near_timedelta'] = 1000000
    df.loc[df.uc_nearhour_show_delta.notnull(),'uc_nearhour_show_delta'] = -99999
    return df

# 划分训练集和测试集
def trainTestSplit(df, splitDate=pd.to_datetime('2018-09-23'), trainPeriod=3, testPeriod=1):
    trainDf = df[(df.context_timestamp<splitDate)&(df.context_timestamp>=splitDate-timedelta(days=trainPeriod))]
    testDf = df[(df.context_timestamp>=splitDate)&(df.context_timestamp<splitDate+timedelta(days=testPeriod))]
    return (trainDf, testDf)

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
        clf.trainCV(kfTrainX, kfTrainY, verbose=False, weight=kfWeight)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
    oofTest[:] = oofTestSkf.mean(axis=1)
    return oofTrain, oofTest


def main():
    # 准备数据
    startTime = datetime.now()
    df = importDf('../data/train_fea_special_sample.csv')
    # df = importDf('../data/train_fea_special.csv')
    df['dataset'] = 0
    predictDf = df.sample(frac=0.1)
    # predictDf = importDf('../data/test_b_fea.csv')
    predictDf['dataset'] = -2
    originDf = pd.concat([df,predictDf], ignore_index=True)
    print('prepare dataset time:', datetime.now()-startTime)

    # 特征处理
    startTime = datetime.now()
    originDf = dataCleaning(originDf)
    originDf = feaFactory(originDf)
    # idx = originDf.date=='2018-09-07'
    # originDf.loc[idx,'weight'] = 1
    # originDf.loc[~idx, 'weight'] = 1
    df = originDf[(originDf['dataset']>=0)]
    predictDf = originDf[originDf['dataset']==-2]

    # 特征筛选
    fea = [
        'item_sales_level','item_price_level','item_collected_level','item_pv_level',#'item_city_id','item_category1',
        'user_gender_id','user_age_level','user_occupation_id','user_star_level',
        'context_page_id',#'hour',
        'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description','shop_star_level','shop_review_num_level',

        'user_his_trade_ratio','user_last_show_timedelta','user_his_show','user_lasthour_show','user_lastdate_show','user_lastdate_trade_ratio',#'user_his_trade',
        'item_his_trade_ratio','item_prop_num','item_lastdate_trade','item_his_trade','item_lastdate_trade_ratio','item_lastdate_trade_delta','item_lastdate_trade_ratio_delta','item_lastdate_show_delta','item_lastdate_show',#item_his_show','item_his_show_delta','item_his_show_perday','item_his_trade_perday','item_his_show_ratio','item_his_show_delta',
        'item_age_delta',#'item_age_std','item_age_mean',
        'item_sales_delta','item_collected_delta','item_pv_delta',#'item_price_delta',
        'shop_his_trade_ratio','shop_lastdate_trade','shop_lastdate_trade_ratio','shop_lastdate_show_delta',#'shop_his_show_ratio',#'shop_his_trade','shop_his_show','shop_his_show_delta','shop_his_show_perday','shop_his_show_perday','shop_his_trade_delta',
        'shop_age_delta',#'shop_age_std','shop_age_mean',
        # 'shop_gender_ratio',# 'shop_gender_ratio_delta',#
        'shop_item_count',# 'shop_collected_mean','shop_price_mean','shop_pv_mean','shop_sales_mean',#'shop_collected_sum','shop_pv_sum','shop_sales_sum','shop_price_sum',
        'shop_sales_delta','shop_pv_delta','shop_collected_delta',#'shop_price_delta',
        # 'shop_review_positive_delta','shop_score_service_delta','shop_score_delivery_delta',# 'shop_score_description_delta',#
        'shop_review_num_delta','shop_star_level_delta','shop_item_count_delta',
        'cate_his_trade_ratio','cate2_his_trade_ratio',#'cate_his_trade','cate_his_show','cate_his_show_perday',
        'cate_age_delta',#'cate_age_std','cate_age_mean',
        # 'cate_gender_ratio_delta',#'cate_gender_ratio',
        'cate_sales_mean',#'cate_sales_sum','cate_collected_mean','cate_collected_sum','cate_price_sum','cate_pv_mean','cate_pv_sum','cate_item_count','cate_price_mean',
        'brand_his_trade_ratio','brand_lastdate_trade_ratio',#'brand_his_trade','brand_his_show','brand_his_show_perday',
        'brand_price_delta','brand_collected_delta','brand_sales_delta','brand_pv_delta',
        'brand_age_delta',
        'shop_brand_item_ratio','shop_brand_count_ratio','shop_brand_special_degree',
        'hour_trade_ratio','predict_cate_num','cate_intersect_num','prop_intersect_num','predict_prop_num',#'prop_jaccard','prop_jaccard_bias',

        'ui_last_show_timedelta','ui_lasthour_show','ui_lastdate_show','ui_lasthour_show_ratio',#'ui_lastdate_trade',
        'uc_his_trade_ratio',#'uc_his_trade',# 'uc_last_show_timedelta',# 'uc_his_trade','uc_his_show','uc_his_show_ratio','uc_lasthour_show_ratio','uc_lasthour_show',
        'uc_price_mean','uc_price_delta',
        'us_last_show_timedelta','us_lasthour_show','us_lasthour_show_ratio',
        'ca_his_trade_ratio','cg_his_trade_ratio',
        'ia_his_trade_ratio','ia_his_trade_delta',#'ia_his_show_delta','ia_his_show_ratio',
        'ig_his_trade_ratio','ig_his_trade_delta',#'ig_his_show_delta','ig_his_show_ratio',
        'ba_his_trade_ratio','ba_his_trade_delta',#'ia_his_show_delta','ia_his_show_ratio',
        'bg_his_trade_ratio','bg_his_trade_delta','bg_his_show_delta',#'ig_his_show_delta','ig_his_show_ratio',

        'user_next_show_timedelta','user_nexthour_show','user_near_timedelta','user_nearhour_show_delta',
        'uc_nearhour_show_delta','uc_near_timedelta','uc_next_show_timedelta','uc_nexthour_show',
        'ui_next_show_timedelta','ui_nexthour_show','ui_near_timedelta',#'ui_nexthour_show_ratio',#'ui_nearhour_show_delta',
        'us_next_show_timedelta','us_nearhour_show_delta','us_nexthour_show',
    ]
    print(df[fea].info())
    print(predictDf[fea].info())
    # exit()

    # 正式模型
    modelName = "xgboost2B"
    startTime = datetime.now()
    xgbModel = XgbModel(feaNames=fea)
    xgbModel.trainCV(df[fea].values, df['is_trade'].values)#, weight=df['weight'].values
    xgbModel.getFeaScore(show=True)
    xgbModel.clf.save_model('%s.model'%modelName)
    print('training time: ', datetime.now()-startTime)

    # 开始预测
    predictDf.loc[:,'predicted_score'] = xgbModel.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['instance_id','predicted_score']].head())
    print('预测均值：', predictDf['predicted_score'].mean())
    # exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
    # exportResult(predictDf[['instance_id','predicted_score','hour']], "%s_hashour.txt" % modelName)

    # 生成stacking数据集
    df['predicted_score'] = np.nan
    predictDf['predicted_score'] = np.nan
    df.loc[:,'predicted_score'], predictDf.loc[:,'predicted_score'] = getOof(xgbModel, df[fea].values, df['is_trade'].values, predictDf[fea].values, stratify=True)#, weight=df['weight'].values
    print('oof training time: ', datetime.now()-startTime)
    xgbModel.getFeaScore(show=True)
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

if __name__ == '__main__':
    main()
