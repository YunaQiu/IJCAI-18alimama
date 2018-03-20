#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
模型： xgboostCV
模型参数：'objective': 'binary:logistic',
        'eval_metric':'logloss',
        'silent': True,
        'eta': 0.1,
        'max_depth': 4,
        'gamma': 0.01,
        'subsample':0.7,
        'colsample_bytree': 0.75,
        'min_child_weight': 1,
        'max_delta_step': 2,
        'num_boost_round': 1500
        'early_stopping_rounds': 2
        'nfold': 5
特征筛选：计算特征对标签的F值， 取得分超过100的特征
特征： 3个商品一级类目，1个商品二级类目
      3个城市id
      商品销量等级/商品收藏等级
      是否女性用户/是否男性用户/用户年龄等级/是否0星用户
      店铺好评率/店铺服务评分/店铺物流评分/店铺描述评分
      商品价格等级3～8
结果： A榜（0.08306）

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
import json

from sklearn.preprocessing import *
import xgboost as xgb
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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

# 转化数据集字段格式，并去重
def formatDf(df):
    strCols = ['instance_id','item_id','user_id','context_id','shop_id','item_brand_id','item_city_id','user_gender_id','user_occupation_id']
    df[strCols] = df[strCols].astype('str')
    df = df.applymap(lambda x: np.nan if (x==-1)|(x=='-1') else x)
    df.drop_duplicates(inplace=True)
    df['context_timestamp'] = pd.to_datetime(df.context_timestamp, unit='s')
    return df

# 计算单特征与标签的F值
def getFeaScore(X, y, feaNames):
    resultDf = pd.DataFrame(index=feaNames)
    selecter = SelectKBest(f_classif, 'all').fit(X, y)
    resultDf['scores'] = selecter.scores_
    resultDf['p_values'] = selecter.pvalues_
    return resultDf

# 将数据集拆分成多个维度的子数据集
def getSubDf(df):
    startTime = datetime.now()
    # 商品数据集
    itemDf = df.drop_duplicates(subset=['item_id'])[['item_id', 'item_category_list', 'item_property_list','item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level','item_collected_level', 'item_pv_level','shop_id']]
    itemDf['item_category_list'] = itemDf[itemDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x.split(';'))
    itemDf['item_category0'] = itemDf[itemDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[0])
    itemDf['item_category1'] = itemDf[itemDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[1] if len(x)>1 else np.nan)
    itemDf['item_category2'] = itemDf[itemDf.item_category_list.notnull()]['item_category_list'].map(lambda x: x[2] if len(x)>2 else np.nan)
    itemDf['item_property_list'] = itemDf[itemDf.item_property_list.notnull()]['item_property_list'].map(lambda x: x.split(';'))

    # 用户数据集
    userDf = df.drop_duplicates(subset=['user_id'])[['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']]

    # 店铺数据集
    shopDf = df.drop_duplicates(subset=['shop_id'])[['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']]

    # 从总数据集中剔除子数据集信息
    df.drop(['item_category_list', 'item_property_list','item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level','item_collected_level', 'item_pv_level',
        'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
        'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description'], axis=1, inplace=True)
    df['predict_category_property'] = df[df.predict_category_property.notnull()]['predict_category_property'].map(
        lambda x: {kv.split(':')[0]:(kv.split(':')[1].split(',') if kv.split(':')[1]!='-1' else []) for kv in x.split(';')})
    df['user_item'] = df['item_id'].astype('str') + '_' + df['user_id'].astype('str')
    return (df, itemDf, userDf, shopDf)

# 将子数据集合并至总数据集
def combineDf(df, itemDf, userDf, shopDf):
    df = df.merge(itemDf.drop(['shop_id'],axis=1), how='left', on=['item_id'])
    df = df.merge(userDf, how='left', on=['user_id'])
    df = df.merge(shopDf, how='left', on=['shop_id'])
    return df

# 添加时间特征
def addTimeFea(df):
    df['hour'] = df.context_timestamp.dt.hour
    df['date'] = df.context_timestamp.dt.date
    return df

class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric':'logloss',
            'silent': True,
            'eta': 0.1,
            'max_depth': 4,
            'gamma': 0.01,
            'subsample':0.7,
            'colsample_bytree': 0.75,
            'min_child_weight': 1,
            'max_delta_step': 2,
            # 'alpha':1600
        }
        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
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

    def trainCV(self, X, y, nFold=5, verbose=True, num_boost_round=1500, early_stopping_rounds=2):
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

    def gridSearch(self, X, y, nFold=5, verbose=1, num_boost_round=70, early_stopping_rounds=30):
        paramsGrids = {
            # 'n_estimators': [50+5*i for i in range(0,30)],
            # 'gamma': [0+10*i for i in range(0,10)],
            # 'max_depth': list(range(3,10)),
            # 'min_child_weight': list(range(1,10))
            # 'subsample': [1-0.05*i for i in range(2,12)]
            # 'colsample_bytree': [1-0.05*i for i in range(0,8)]
            # 'reg_alpha':[1000+100*i for i in range(0,20)]
            # 'max_delta_step': [0+1*i for i in range(0,10)]
        }
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
                # reg_alpha = self.params['alpha'],
                n_estimators = num_boost_round
            ),
            param_grid = paramsGrids,
            scoring = 'neg_log_loss',
            cv = nFold,
            verbose = verbose,
            n_jobs = 3
        )
        gsearch.fit(X, y)
        print(pd.DataFrame(gsearch.cv_results_))
        print(gsearch.best_params_)
        exit()

    def predict(self, X):
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
def feaFactory(df):
    startTime = datetime.now()
    df = formatDf(df)
    df, itemDf, userDf, shopDf = getSubDf(df)
    df = addTimeFea(df)
    df = combineDf(df, itemDf, userDf, shopDf)
    df = addOneHot(df, ['item_category1','item_category2','item_price_level','user_gender_id','user_star_level','user_age_level','user_occupation_id','item_city_id'])
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
def getOof(clf, trainX, trainY, testX, nFold=10):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        clf.trainCV(kfTrainX, kfTrainY, verbose=False)
        oofTrain[testIdx] = clf.predict(kfTestX)
    clf.trainCV(trainX,trainY, verbose=False)
    oofTest = clf.predict(testX)
    return oofTrain, oofTest


if __name__ == '__main__':
    # 导入数据
    df = importDf('../data/round1_ijcai_18_train_20180301.txt')

    # 特征处理
    df = feaFactory(df)

    # 特征筛选
    # tempCol = ['item_price_level_%s'%x for x in set(df['item_price_level'].dropna().values)]
    # resultDf = getFeaScore(df[tempCol].values, df['is_trade'].values, tempCol)
    # print(resultDf[resultDf.scores>10])
    fea = [
        'item_category1_7258015885215914736','item_category1_8277336076276184272','item_category1_2642175453151805566','item_category2_8868887661186419229','item_city_id_3819392654129628501','item_city_id_7534238860363577544','item_city_id_5918626470536001929',
        'item_sales_level','item_collected_level',
        'user_gender_id_0','user_gender_id_1',
        'user_age_level','user_star_level_3000.0',
        'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description'
    ]
    fea.extend(['item_price_level_%s'%x for x in range(3,9)])
    # exit()

    # 测试模型效果
    costDf = pd.DataFrame(index=fea+['cost'])
    xgbModel = XgbModel(feaNames=fea)
    for dt in pd.date_range(start='2018-09-21', end='2018-09-23', freq='D'):
        trainDf, testDf = trainTestSplit(df, dt)
        # xgbModel.gridSearch(trainDf[fea].values, trainDf['is_trade'].values, nFold=5)
        xgbModel.trainCV(trainDf[fea].values, trainDf['is_trade'].values, nFold=5)
        testDf.loc[:,'predict'] = xgbModel.predict(testDf[fea].values)
        scoreDf = xgbModel.getFeaScore()
        scoreDf.columns = [dt.strftime('%Y-%m-%d')]
        costDf = costDf.merge(scoreDf, how='left', left_index=True, right_index=True)
        cost = metrics.log_loss(testDf['is_trade'].values, testDf['predict'].values)
        costDf.loc['cost',dt.strftime('%Y-%m-%d')] = cost
    print(costDf)
    exit()

    # 正式模型
    modelName = "xgboost1A"
    xgbModel.trainCV(df[fea].values, df['is_trade'].values)
    xgbModel.getFeaScore(show=True)
    xgbModel.clf.save_model('%s.model'%modelName)

    # 预测集准备
    startTime = datetime.now()
    predictDf = importDf('../data/round1_ijcai_18_test_a_20180301.txt')
    predictDf = feaFactory(predictDf)
    # 填补缺失字段
    print('缺失字段：', [x for x in fea if x not in predictDf.columns])
    for x in [x for x in fea if x not in predictDf.columns]:
        predictDf[x] = 0
    print("预测集：\n",predictDf.head())
    print(predictDf[fea].info())

    # 开始预测
    predictDf.loc[:,'predicted_score'] = xgbModel.predict(predictDf[fea].values)
    print("预测结果：\n",predictDf[['instance_id','predicted_score']].head())
    # exportResult(predictDf, "%s_predict.csv" % modelName)
    exportResult(predictDf[['instance_id','predicted_score']], "%s.txt" % modelName)
