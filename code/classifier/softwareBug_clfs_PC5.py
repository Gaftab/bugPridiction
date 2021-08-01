

import numpy
# import numpy as np
import scipy
import pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
# from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler ,LabelBinarizer
pandas.options.mode.chained_assignment = None 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
#0) INITIALIZE DATASET

csv = r'/content/drive/MyDrive/softwareBugPred/bugPridiction/data/pc5.csv'
dataset = pandas.read_csv(csv)
X=dataset.iloc[:,0:37]
le= preprocessing.LabelEncoder()
dataset['Class']=le.fit_transform(dataset['Class'].astype(str))
dataset['Class']= dataset['Class'].astype('bool')
# print(dataset['Class'].unique())
# dataset['Class']=pandas.get_dummies(dataset['Class'])

# #1) FEATURE ENGINEERING
# #1.1) Normalization 

# #The min-max normalization was applied to the numerical features of the dataset to remove instability.

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# norm_df= pandas.DataFrame(norm_feature_cols, columns=dataset.columns)
# x_sm=train=norm_df
x_sm=train=X

# # # #Converting data frame to sparce matrix
x_sm=scipy.sparse.csr_matrix(x_sm)
y=dataset.iloc[:,-1]

# 1.3) Feature Selection 

# Feature selection  using a chi-square score was applied  for each applied machine learning algorithm. 

# COMMENT OUT following code block for experimenting different feature sizes for each classifier
#1
clf=svm.SVC()
for x in range(1,37,11):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_sm, y)
    x_t= fit.transform(x_sm)
    scores = cross_val_score(clf, x_sm, y, cv=10 )
    print('svm cross validation scores ',scores)

# #2
# clf=LogisticRegression()
# for x in range(1,21,11):
#     test = SelectKBest(score_func=chi2, k=x)
#     fit = test.fit(x_sm, y)
#     x_t= fit.transform(x_sm)
#     scores = cross_val_score(clf, x_sm, y, cv=10)
#     print('LogisticRegression cross validation scores ',scores)
# #3
# clf=RandomForestClassifier()
# for x in range(1,21,11):
#     test = SelectKBest(score_func=chi2, k=x)
#     fit = test.fit(x_sm, y)
#     x_t= fit.transform(x_sm)
#     scores = cross_val_score(clf, x_sm, y, cv=10)
#     print('RandomForestClassifier cross validation scores ',scores)
# #4
# GradientBoostingClassifier
# clf=GradientBoostingClassifier()
# for x in range(1,21,11):
#     test = SelectKBest(score_func=chi2, k=x)
#     fit = test.fit(x_sm, y)
#     x_t= fit.transform(x_sm)
#     scores = cross_val_score(clf, x_sm, y, cv=10)
#     print(' GradientBoostingClassifier cross validation scores ',scores)
# #5
# clf=AdaBoostClassifier()
# for x in range(1,21, 11):
#     test = SelectKBest(score_func=chi2, k=x)
#     fit = test.fit(x_sm, y)
#     scores = cross_val_score(clf, x_sm, y, cv=10)
#     print('AdaBoostClassifier cross validation scores ',scores)
# #6
# clf=RandomForestClassifier()
# for x in range(1,21,11):
#     test = SelectKBest(score_func=chi2, k=x)
#     fit = test.fit(x_sm, y)
#     x_t= fit.transform(x_sm)
#     scores = cross_val_score(clf, x_sm, y, cv=10)
#     print('RandomForestClassifier cross validation scores ',scores)


# # Use k that has the most highest scores.
test = SelectKBest(score_func=chi2, k=17)
fit = test.fit(x_sm, y)

x_ts=x_sm

# #2) PARAMETER OPTIMIZATION 

# #Grid search was applied to the used machine learning algorithms (except NBM) . 
# #Note that it takes a few days. You can skip this step, the parameters are predefined in third step.
# #COMMENT OUT the related code blocks for experimenting parameter optimization of classifiers.

# #2.1) SVM 

# #Experiment both x; x_ts

x=x_ts
# scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

# search_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#                    ]
# search = GridSearchCV(SVC(), search_grid, cv=10, n_jobs=16 ,scoring='%s_macro' % 'recall' )
# search.fit(x, y)
# search.best_params_
# svm_best_par=search.best_params_
# print('SVM best parameters ',svm_best_par)

# #2.3)GradientBoostingClassifier

# search_grid = {"loss":["deviance"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "max_features":["log2","sqrt"],
#     "n_estimators":[100],
#      "max_depth":[3,5,8]}

# search = GridSearchCV(GradientBoostingClassifier(), search_grid, cv=10,n_jobs=16 ,scoring='%s_macro' % 'recall')
# search.fit(x, y)
# search.best_params_
# bagRg_best_par=search.best_params_
# print('GradientBoostingClassifier best parameters ',bagRg_best_par)



# # #2.4) Logistic Regression

# search_grid={"C":numpy.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
# search=GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=1000), search_grid, cv=10)
# search.fit(x,y)
# search.best_params_
# print('Logistic Regression Best Parameters ',search.best_params_)


# # # #2.5) AdaboostClassifier


# search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,0.1]}
# search=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=search_grid,scoring='recall' , cv=10, n_jobs=32)
# search.fit(x,y)
# search.best_params_
# ada_best_par=search.best_params_
# print('AdaboostClassifier Best Parameters ',ada_best_par)


# # # #2.6) RandomForestClassifier

# search_grid = { 
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=search_grid, cv= 10, n_jobs=-1)
# search.fit(x, y)
# search.best_params_
# Rf_best_par=search.best_params_
# print('RandomForestClassifier Best Parameters ',Rf_best_par)

# # #2.7) ExtraTreesClassifier

# search_grid = { 
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2'],
#      'min_weight_fraction_leaf':[0.1],
#     'criterion':['gini']
# }
# search = GridSearchCV(ExtraTreesClassifier(), param_grid=search_grid, cv= 10, n_jobs=-1)
# search.fit(x, y)
# search.best_params_
# ETC_best_par=search.best_params_
# print('ExtraTreesClassifier Best Parameters ',ETC_best_par)

#3) CLASSIFIERS
    
#Parameters of a classifier on related dataset obtained from second step.

# #3) CLASSIFIERS

# 3.1) SVM
clf=svm.SVC(C=1000, gamma=0.01,kernel="rbf")
svm_scores_pre = cross_val_score(clf, x_ts, y, cv=10,scoring='precision_weighted') 
svm_scores_acc = cross_val_score(clf, x_ts, y, cv=10,scoring='accuracy') 
svm_scores_roc = cross_val_score(clf, x_ts, y, cv=10,scoring='roc_auc') 
svm_scores_f1 = cross_val_score(clf, x_ts, y, cv=10,scoring='f1_weighted') 
svm_scores_recall = cross_val_score(clf, x_ts, y, cv=10,scoring='recall_weighted') 
svm_acc=svm_scores_acc.mean()
svm_pre=svm_scores_pre.mean()
svm_roc=svm_scores_roc.mean()
svm_f1=svm_scores_f1.mean()
svm_recall=svm_scores_recall.mean()

# # #3.2) NBM
  
clf= MultinomialNB()
nb_scores_pre = cross_val_score(clf, x_ts, y, cv=10,scoring='precision_weighted') 
nb_scores_acc = cross_val_score(clf, x_ts, y, cv=10,scoring='accuracy') 
nb_scores_roc = cross_val_score(clf, x_ts, y, cv=10,scoring='roc_auc') 
nb_scores_f1 = cross_val_score(clf, x_ts, y, cv=10,scoring='f1_weighted') 
nb_scores_recall = cross_val_score(clf, x_ts, y, cv=10,scoring='recall_weighted') 
NB_acc=nb_scores_acc.mean()
NB_pre=nb_scores_pre.mean()
NB_roc=nb_scores_roc.mean()
NB_f1=nb_scores_f1.mean()
NB_recall=nb_scores_recall.mean()

# #3.3) Logistic Regression

clf=LogisticRegression(C=0.1,penalty="l2",solver='lbfgs', max_iter=1000)
LR_scores_pre = cross_val_score(clf, x_ts, y, cv=10,scoring='precision_weighted') 
LR_scores_acc = cross_val_score(clf, x_ts, y, cv=10,scoring='accuracy') 
LR_scores_roc = cross_val_score(clf, x_ts, y, cv=10,scoring='roc_auc') 
LR_scores_f1 = cross_val_score(clf, x_ts, y, cv=10,scoring='f1_weighted') 
LR_scores_recall = cross_val_score(clf, x_ts, y, cv=10,scoring='recall_weighted') 
LR_acc=LR_scores_acc.mean()
LR_pre=LR_scores_pre.mean()
LR_roc=LR_scores_roc.mean()
LR_f1=LR_scores_f1.mean()
LR_recall=LR_scores_recall.mean()

# #3.4) AdaBoost

clf=AdaBoostClassifier(learning_rate=0.1, n_estimators=2000)
AdaBoost_scores_pre = cross_val_score(clf, x_ts, y, cv=10,scoring='precision_weighted') 
AdaBoost_scores_acc = cross_val_score(clf, x_ts, y, cv=10,scoring='accuracy') 
AdaBoost_scores_roc = cross_val_score(clf, x_ts, y, cv=10,scoring='roc_auc') 
AdaBoost_scores_f1 = cross_val_score(clf, x_ts, y, cv=10,scoring='f1_weighted') 
AdaBoost_scores_recall = cross_val_score(clf, x_ts, y, cv=10,scoring='recall_weighted') 
AdaBoost_acc=AdaBoost_scores_acc.mean()
AdaBoost_pre=AdaBoost_scores_pre.mean()
AdaBoost_roc=AdaBoost_scores_roc.mean()
AdaBoost_f1=AdaBoost_scores_f1.mean()
AdaBoost_recall=AdaBoost_scores_recall.mean()
# print(AdaBoost_acc)
# print(AdaBoost_pre)
# print(AdaBoost_recall)
# print(AdaBoost_roc)
# print(AdaBoost_f1)

# # #3.5) RF
 
clf=RandomForestClassifier( max_features= 'sqrt', n_estimators= 700)
RF_scores_pre = cross_val_score(clf, x_ts, y, cv=10,scoring='precision_weighted') 
RF_scores_acc = cross_val_score(clf, x_ts, y, cv=10,scoring='accuracy') 
RF_scores_roc = cross_val_score(clf, x_ts, y, cv=10,scoring='roc_auc') 
RF_scores_f1 = cross_val_score(clf, x_ts, y, cv=10,scoring='f1_weighted') 
RF_scores_recall = cross_val_score(clf, x_ts, y, cv=10,scoring='recall_weighted') 
RF_acc=RF_scores_acc.mean()
RF_pre=RF_scores_pre.mean()
RF_roc=RF_scores_roc.mean()
RF_f1=RF_scores_f1.mean()
RF_recall=RF_scores_recall.mean()

# #3.6) GradientBoostingClassifier

clf=GradientBoostingClassifier(loss='deviance', max_features='sqrt',n_estimators=100, learning_rate=0.2, max_depth=8, random_state=0)
GBC_scores_pre = cross_val_score(clf, x_ts, y, cv=10,scoring='precision_weighted') 
GBC_scores_acc = cross_val_score(clf, x_ts, y, cv=10,scoring='accuracy') 
GBC_scores_roc = cross_val_score(clf, x_ts, y, cv=10,scoring='roc_auc') 
GBC_scores_f1 = cross_val_score(clf, x_ts, y, cv=10,scoring='f1_weighted') 
GBC_scores_recall = cross_val_score(clf, x_ts, y, cv=10,scoring='recall_weighted') 
GBC_acc=GBC_scores_acc.mean()
GBC_pre=GBC_scores_pre.mean()
GBC_roc=GBC_scores_roc.mean()
GBC_f1=GBC_scores_f1.mean()
GBC_recall=GBC_scores_recall.mean()

# # #3.7)ExtraTreesClassifier
 
clf= ExtraTreesClassifier(n_estimators=200, max_features='auto', random_state=0, min_weight_fraction_leaf=0.1, criterion='gini')
ETC_scores_pre = cross_val_score(clf, x_ts, y, cv=10,scoring='precision_weighted') 
ETC_scores_acc = cross_val_score(clf, x_ts, y, cv=10,scoring='accuracy') 
ETC_scores_roc = cross_val_score(clf, x_ts, y, cv=10,scoring='roc_auc') 
ETC_scores_f1 = cross_val_score(clf, x_ts, y, cv=10,scoring='f1_weighted') 
ETC_scores_recall = cross_val_score(clf, x_ts, y, cv=10,scoring='recall_weighted') 
ETC_acc=ETC_scores_acc.mean()
ETC_pre=ETC_scores_pre.mean()
ETC_roc=ETC_scores_roc.mean()
ETC_f1=ETC_scores_f1.mean()
ETC_recall=ETC_scores_recall.mean()



#4) RESULTS

labels = ['SVM','NBM','LR','AdaBoost','RF','GBC','ETC']
pre = [svm_pre,NB_pre,LR_pre,AdaBoost_pre,RF_pre,GBC_pre,ETC_pre]
re = [svm_recall,NB_recall,LR_recall,AdaBoost_recall,RF_recall,GBC_recall,ETC_recall]
aoc = [svm_roc,NB_roc,LR_roc,AdaBoost_roc,RF_roc,GBC_roc,ETC_roc]
f1 = [svm_f1,NB_f1,LR_f1,AdaBoost_f1,RF_f1,GBC_f1,ETC_f1]
acurracy = [svm_acc,NB_acc,LR_acc,AdaBoost_acc,RF_acc,GBC_acc,ETC_acc] 

x = numpy.arange(len(labels))  # the label locations
width = 0.17  # the width of the bars

fig = plt.figure(figsize=(30,22))
ax = fig.add_subplot()
rects1 = ax.bar(x - width*2, acurracy, width, label='Acurracy score')
rects2 = ax.bar(x - width, pre, width, label='Precision score')
rects3 = ax.bar(x, re, width, label='Recall score')
rects4 = ax.bar(x + width*2, aoc, width, label='ROC score')
rects5 = ax.bar(x + width, f1, width, label='F1 score')




# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage Score')
ax.set_title('Results of the experimented Classifiers for PC5 dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim(0.50, 0.99)
ax.legend(loc="upper center", bbox_to_anchor=(0.4, 1.20),prop={"size":20})

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
    
        ax.annotate('{}'.format(round(100*height,2)),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                     fontsize=14,
               
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

plt.show()









# labels = ['SVM','NBM','LR','AdaBoost','RF','GBC','ETC']
# all_scores = [svm_score,nbm_score,lg_score,adaBoost_score,rf_score,GrB_score,etc_score]

# x = numpy.arange(len(labels))  # the label locations
# width = 0.5  # the width of the bars

# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot()
# rects1 = ax.bar(x, all_scores, width)


# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy of the experimented Classifiers for CMI dataset')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# plt.ylim(0.80, 0.99)
# # ax.legend(loc="upper center", bbox_to_anchor=(0.4, 1.20))

# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
    
#         ax.annotate('{}'.format(round(100*height,3)),
#                     xy=(rect.get_x() + rect.get_width()/2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
               
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# autolabel(rects1)
# plt.show()
# result_dir = '/content/drive/MyDrive/softwareBugPred/bugPridiction/results'
# plt.savefig(f"{result_dir}/test.png")





