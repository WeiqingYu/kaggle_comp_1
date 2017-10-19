
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
lab = train['IsBadBuy']
ntrain = len(train)
train = train.drop('IsBadBuy',axis=1)
train = train.drop('RefId',axis=1)
test = test.drop('RefId',axis=1)
train.head()

print('Ratio of kicked case:', sum(lab==1)/len(lab))
datcheck = train.loc[:,train.dtypes!=object].sample(1000).values
datcheck[np.isnan(datcheck)]=0
cor1 = np.corrcoef(datcheck.transpose())
print(train.columns[train.dtypes!=object])
plt.imshow(cor1, cmap='hot', interpolation='nearest')
plt.show()
train.isnull().sum()

train['PurchDate'] = pd.to_datetime(train['PurchDate']).values.astype(np.int64)/1000000000000
test['PurchDate'] = pd.to_datetime(test['PurchDate']).values.astype(np.int64)/1000000000000
aldat = pd.concat([train,test],ignore_index=True)
aldat = pd.get_dummies(aldat)
train = aldat.iloc[:ntrain,].values
test = aldat.iloc[ntrain:,].values
train[np.isnan(train)] = 0
test[np.isnan(test)] = 0
train.shape
pcamodel = PCA(n_components=40)
pcamodel.fit(train)
train = pcamodel.transform(train)
test = pcamodel.transform(test)
samp = np.random.rand(ntrain)<0.97
train, lab1, cross, lab2 = train[samp], lab[samp], train[~samp], lab[~samp]
xgtest = xgb.DMatrix(test)
xgtrain = xgb.DMatrix(train,label=lab1)
xgeval = xgb.DMatrix(cross,label=lab2)
evallist = [(xgtrain,'train'),(xgeval,'val')]

param = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'binary:logistic',
         'subsample':0.3,'lambda':1.5,'alpha':1, 'eval_metric':['auc','error']}
xgmodel = xgb.train(param,xgtrain,num_boost_round=30,evals=evallist)
pred = xgmodel.predict(xgeval)
lab2.index=range(len(lab2))
xpred = pred>0.5
print(sum(np.logical_and(xpred==1,lab2==1)),sum(np.logical_and(xpred==1,lab2==0)))
print(sum(np.logical_and(xpred==0,lab2==1)),sum(np.logical_and(xpred==0,lab2==0)))
