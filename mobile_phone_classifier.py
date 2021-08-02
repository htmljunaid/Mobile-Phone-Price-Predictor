# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:06:42 2021

@author: Muhammad Junaid Hanif
"""

#%% Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sklearn
from sklearn.tree import DecisionTreeClassifier
from dmba import plotDecisionTree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

#%% Setting up Data

data = pd.read_csv('E:\Python\mobile price classification/train.csv')
data.info()

#%% assigning variables

y=data['price_range']

X = data.drop(['price_range'],axis=1)

#%% Visualizing Data

data.info()

data.describe()


fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

sns.boxplot(x="price_range", y="battery_power", data=data)

sns.boxplot(x="price_range", y="int_memory", data=data)

sns.boxplot(x="price_range", y="ram", data=data)

plt.figure(figsize=(10,6))
data['fc'].hist(alpha=0.5,color='blue',label='Front camera')
data['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')

sns.pairplot(data,hue='price_range')

sns.jointplot(x='ram',y='price_range',data=data,color='red',kind='kde')

sns.pointplot(y="int_memory", x="price_range", data=data)

#%% K Nearest Neighbors

#%% Partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#%% Normalization

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
XNtrain = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
XNtest = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

#%% Finding The best k?

resList=[]
for k in range(1,25):
    knn_bk = KNeighborsClassifier(n_neighbors=k)
    knn_bk.fit(XNtrain,y_train)
    y_pred = knn_bk.predict(XNtest)
    acc = metrics.accuracy_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred, pos_label=1)
    resList.append([k,acc,rec])
colsRes = ['k','Accuracy','Recall_Pos']
results = pd.DataFrame(resList, columns=colsRes)
print(results)

#%% k=3 train and test

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(XNtrain,y_train)

y_knn = knn.predict(XNtest)

#%% Evaluating kNN

kappa = metrics.cohen_kappa_score(y_test,y_knn)
print(kappa)

#Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_knn)
cmPlot = metrics.ConfusionMatrixDisplay(cm)
cmPlot.plot(cmap='YlOrRd')
cmPlot.figure_.savefig('figs/CM_kNN.png')

#Classification Report
creport_KNN = metrics.classification_report(y_test, y_knn)
print(creport_KNN)


#%%   Cart model

#%% The Best Depth?
#train
ctree_full = DecisionTreeClassifier(random_state=18)
ctree_full.fit(X_train, y_train)
plotDecisionTree(ctree_full, feature_names = X.columns)

# parameter setting
param_grid = {'max_depth':[1,2,3,4,5],
              'min_samples_split':[5,10,15,20],
              'min_impurity_decrease':[0, 0.0005, 0.001, 0.05, 0.01] }

gridSearch = GridSearchCV(ctree_full,
                          param_grid,
                          scoring = 'recall',
                          n_jobs=-1)

gridSearch.fit(X_train,y_train)
print("Best Recall:", gridSearch.best_score_)

print("Best parameters:", gridSearch.best_params_)

ctree_best = gridSearch.best_estimator_
plotDecisionTree(ctree_best, feature_names = X.columns)
#Best depth= 5

ctree = DecisionTreeClassifier(random_state=1, max_depth=3)
ctree.fit(X_train, y_train)
plotDecisionTree(ctree,
                 feature_names = X.columns,
                 class_names=ctree.classes_)

#%% TEST CART Model

y_ctree = ctree.predict(X_test)

#%% Evaluating CART Model

kappa = metrics.cohen_kappa_score(y_test,y_ctree)
print(kappa)
#kappa is 0.606

#Confusion Matrix
cm_CART = metrics.confusion_matrix(y_test, y_ctree)
cmPlot_CART = metrics.ConfusionMatrixDisplay(cm_CART)
CART_plot=cmPlot_CART.plot(cmap='YlOrRd')
CART_plot.figure_.savefig('FIGS/CM_CART.png')

#Classification Report
creport_CART = metrics.classification_report(y_test, y_ctree)
print(creport_CART)

#%% Random Forest

rf_18 = RandomForestClassifier(n_estimators=500,random_state=18)
rf_18.fit(X_train,y_train)
importances = rf_18.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_18.estimators_], axis=0)
df = pd.DataFrame({'feature': X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance', ascending=False)
print(df)
ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')
plt.show()

y_rf_18 = rf_18.predict(X_test)

#%% Evaluating Random Forest

kappa = metrics.cohen_kappa_score(y_test,y_rf_18)
print(kappa)


#Confusion Matrix
cm_RF = metrics.confusion_matrix(y_test, y_rf_18)
cmPlot_RF = metrics.ConfusionMatrixDisplay(cm_RF)
a_RF=cmPlot_RF.plot(cmap='YlOrRd')
a_RF.figure_.savefig('figs/CM_RF.png')
#Classification Report
creport_RF = metrics.classification_report(y_test, y_rf_18)
print(creport_RF)

