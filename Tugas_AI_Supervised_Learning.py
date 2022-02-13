#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

df=pd.read_csv('heart.csv')
X=df.drop('target', axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
model_1 = DecisionTreeRegressor()
param_grid_1={'criterion':['mse','friedman_mse','mae'],'splitter':['best','random'],'max_depth':np.arange(2,12)}
gcsv = GridSearchCV(model_1, param_grid=param_grid_1, scoring='roc_auc',cv=5)
# rscv=RandomizedSearchCV(model_1, param_grid_1, n_iter=20, scoring='roc_auc', cv=5)
gcsv.fit(X_train,y_train)
y_predict_train=gcsv.predict(X_train)
y_predict_test=gcsv.predict(X_test)

print('hyperparameter terbaik model DT :',gcsv.best_params_)
print('score validasi terbaik model DT :',gcsv.best_score_)
valid_score=roc_auc_score(y_test,y_predict_test)
train_score=roc_auc_score(y_train,y_predict_train)

print('score data test :',valid_score)
print('score data training :',train_score)


# In[38]:


model_2 = LogisticRegression()
param_grid_2={'penalty':['l1','l2'],'C':(np.logspace(-4,4,20)),'solver': ['liblinear']}
gcsv_2 = GridSearchCV(model_2, param_grid=param_grid_2, scoring='roc_auc',cv=5)
# rscv_2=RandomizedSearchCV(model_2, param_grid_2, n_iter=20, scoring='roc_auc', cv=5)
gcsv_2.fit(X_train,y_train)
y_predict_train_2 = gcsv_2.predict(X_train)
y_predict_test_2 = gcsv_2.predict(X_test)

print('hyperparameter terbaik model DT :',gcsv_2.best_params_)
print('score validasi terbaik model DT :',gcsv_2.best_score_)
valid_score_2=roc_auc_score(y_test,y_predict_test_2)
train_score_2=roc_auc_score(y_train,y_predict_train_2)

print('score data test :',valid_score_2)
print('score data training :',train_score_2)


# In[53]:


model_3 = RandomForestRegressor()
param_grid_3={'criterion':['mse','mae'],'max_features':['auto','sqrt','log2'],'n_estimators':np.arange(10,100),'max_depth':np.arange(2,12)}
# gcsv_3 = GridSearchCV(model_3, param_grid=param_grid_3, scoring='roc_auc',cv=5)
rscv_3=RandomizedSearchCV(model_3, param_grid_3, n_iter=100, scoring='roc_auc', cv=5)
rscv_3.fit(X_train,y_train)
y_predict_train_3 = rscv_3.predict(X_train)
y_predict_test_3 = rscv_3.predict(X_test)

print('hyperparameter terbaik model DT :',rscv_3.best_params_)
print('score validasi terbaik model DT :',rscv_3.best_score_)
valid_score_3=roc_auc_score(y_test,y_predict_test_3)
train_score_3=roc_auc_score(y_train,y_predict_train_3)

print('score data test :',valid_score_3)
print('score data training :',train_score_3)


# In[16]:


model_4 = Lasso()
param_grid_4={'alpha': (np.logspace(-8, 8, 100)),'fit_intercept':['True','False'],'normalize':['True','False']}
gcsv_4 = GridSearchCV(model_4, param_grid=param_grid_4, scoring='roc_auc',cv=5)
# rscv_4=RandomizedSearchCV(model_4, param_grid_4, n_iter=20, scoring='roc_auc', cv=5)
gcsv_4.fit(X_train,y_train)
y_predict_train_4 = gcsv_4.predict(X_train)
y_predict_test_4 = gcsv_4.predict(X_test)

print('hyperparameter terbaik model DT :',gcsv_4.best_params_)
print('score validasi terbaik model DT :',gcsv_4.best_score_)
valid_score_4=roc_auc_score(y_test,y_predict_test_4)
train_score_4=roc_auc_score(y_train,y_predict_train_4)

print('score data test :',valid_score_4)
print('score data training :',train_score_4)


# In[22]:


model_5 = Ridge()
param_grid_5={'alpha': (np.logspace(-8, 8, 100)),'fit_intercept':['True','False'],'normalize':['True','False']}
gcsv_5 = GridSearchCV(model_5, param_grid=param_grid_5, scoring='roc_auc',cv=5)
# rscv_4=RandomizedSearchCV(model_4, param_grid_4, n_iter=20, scoring='roc_auc', cv=5)
gcsv_5.fit(X_train,y_train)
y_predict_train_5 = gcsv_5.predict(X_train)
y_predict_test_5 = gcsv_5.predict(X_test)

print('hyperparameter terbaik model DT :',gcsv_5.best_params_)
print('score validasi terbaik model DT :',gcsv_5.best_score_)
valid_score_5=roc_auc_score(y_test,y_predict_test_5)
train_score_5=roc_auc_score(y_train,y_predict_train_5)

print('score data test :',valid_score_5)
print('score data training :',train_score_5)


# In[ ]:




