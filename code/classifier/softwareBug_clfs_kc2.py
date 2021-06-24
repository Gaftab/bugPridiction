

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
from sklearn.preprocessing import MinMaxScaler
pandas.options.mode.chained_assignment = None 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

#0) INITIALIZE DATASET

    
csv = r'/content/drive/MyDrive/softwareBugPred/bugPridiction/data/kc2_csv.csv'
dataset = pandas.read_csv(csv)
feature_cols = ['loc','v(g)','ev(g)','iv(g)','n','v','l','d','i','e','b','t','lOCode','lOComment','lOBlank','lOCodeAndComment','uniq_Op','uniq_Opnd','total_Op','total_Opnd','branchCount']
X=dataset[feature_cols]
# print(X)


# #1) FEATURE ENGINEERING

# #1.1) Normalization 

# #The min-max normalization was applied to the numerical features of the dataset to remove instability.

scaler = MinMaxScaler()
X[feature_cols] = scaler.fit_transform(X[feature_cols])

# norm_df= pandas.DataFrame(norm_feature_cols, columns=dataset.columns)
# x_sm=train=norm_df
x_sm=train=X[feature_cols]

# # # #Converting data frame to sparce matrix
x_sm=scipy.sparse.csr_matrix(x_sm.values)
y=dataset.problems
# print(x_sm)


# 1.3) Feature Selection 

# Feature selection  using a chi-square score was applied  for each applied machine learning algorithm to select relevant textual features. 

# COMMENT OUT following code block for experimenting different feature sizes for each classifier
#1
clf=svm.SVC()
for x in range(1,21,11):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_sm, y)
    x_t= fit.transform(x_sm)
    scores = cross_val_score(clf, x_sm, y, cv=10)
    print('svm cross validation scores ',scores)

#2
clf=LogisticRegression()
for x in range(1,21,11):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_sm, y)
    x_t= fit.transform(x_sm)
    scores = cross_val_score(clf, x_sm, y, cv=10)
    print('LogisticRegression cross validation scores ',scores)
#3
clf=RandomForestClassifier()
for x in range(1,21,11):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_sm, y)
    x_t= fit.transform(x_sm)
    scores = cross_val_score(clf, x_sm, y, cv=10)
    print('RandomForestClassifier cross validation scores ',scores)
#4
GradientBoostingClassifier
clf=GradientBoostingClassifier()
for x in range(1,21,11):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_sm, y)
    x_t= fit.transform(x_sm)
    scores = cross_val_score(clf, x_sm, y, cv=10)
    print(' GradientBoostingClassifier cross validation scores ',scores)
#5
clf=AdaBoostClassifier()
for x in range(1,21, 11):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_sm, y)
    scores = cross_val_score(clf, x_sm, y, cv=10)
    print('AdaBoostClassifier cross validation scores ',scores)
#6
clf=RandomForestClassifier()
for x in range(1,21,11):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_sm, y)
    x_t= fit.transform(x_sm)
    scores = cross_val_score(clf, x_sm, y, cv=10)
    print('RandomForestClassifier cross validation scores ',scores)


# # Use k that has the most highest scores.
test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(x_sm, y)

x_ts=x_sm

# #2) PARAMETER OPTIMIZATION 

# #Grid search was applied to the used machine learning algorithms (except NBM) . 
# #Note that it takes a few days. You can skip this step, the parameters are predefined in third step.
# #COMMENT OUT the related code blocks for experimenting parameter optimization of classifiers.

# #2.1) SVM 

# #Experiment both x; x_ts

x=x_ts

# search_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#                    ]
# search = GridSearchCV(SVC(), search_grid, cv=10, n_jobs=16 ,scoring='%s_macro' % 'recall')
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



# #2.2) KNN

# search_grid = dict(n_neighbors = list(range(1,31)), metric = ['euclidean', 'manhattan'] )
# search = GridSearchCV(KNeighborsClassifier(), search_grid, cv = 10, scoring = 'recall', n_jobs=16)
# search.fit(x,y)
# search.best_params_



# #2.4) Logistic Regression

# search_grid={"C":numpy.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
# search=GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=1000), search_grid, cv=10)
# search.fit(x,y)
# search.best_params_
# print('Logistic Regression Best Parameters ',search.best_params_)


# # #2.5) AdaboostClassifier


# search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,0.1]}
# search=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=search_grid,scoring='recall' , cv=10, n_jobs=32)
# search.fit(x,y)
# search.best_params_
# ada_best_par=search.best_params_
# print('AdaboostClassifier Best Parameters ',ada_best_par)


# # #2.6) RandomForestClassifier

# search_grid = { 
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=search_grid, cv= 10, n_jobs=-1)
# search.fit(x, y)
# search.best_params_
# Rf_best_par=search.best_params_
# print('RandomForestClassifier Best Parameters ',Rf_best_par)

# #2.7) ExtraTreesClassifier

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

#3) CLASSIFIERS

# 3.1) SVM
clf=svm.SVC(C=5, kernel="linear")
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
svm_score=scores_ts.mean()
print('SVM Accuracy: ',svm_score)

# #3.2) KNN

# #3.2.A)  text and social media features
# clf= KNeighborsClassifier(n_neighbors= 3, metric="euclidean")
# scores_ts = cross_val_score(clf, x_ts, y, cv=10)
# knnTs=scores_ts.mean()

# #3.2) NBM
  
clf= MultinomialNB()
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
nbm_score=scores_ts.mean()
print('NBM Accuracy: ',nbm_score)



#3.3) Logistic Regression

clf=LogisticRegression(C=100,penalty="l2",solver='lbfgs', max_iter=1000)
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
lg_score=scores_ts.mean()
print('Logistic Regression Accuracy: ',lg_score)

# #3.4) AdaBoost
 
clf=AdaBoostClassifier(learning_rate=0.01, n_estimators=2000)
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
adaBoost_score=scores_ts.mean()
print('AdaBoost Accuracy: ',adaBoost_score)

# #3.5) RF
 
clf=RandomForestClassifier( max_features= 'sqrt', n_estimators= 700)
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
rf_score=scores_ts.mean()
print('RF Accuracy: ',rf_score)

#3.6) GradientBoostingClassifier

clf=GradientBoostingClassifier(loss='deviance', max_features='sqrt',n_estimators=100, learning_rate=0.01, max_depth=3, random_state=0)
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
GrB_score=scores_ts.mean()
print('GradientBoostingClassifier Accuracy: ',GrB_score)

# #3.7)ExtraTreesClassifier
 
clf= ExtraTreesClassifier(n_estimators=200, max_features='log2', random_state=0, min_weight_fraction_leaf=0.1, criterion='gini')
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
etc_score=scores_ts.mean()
print('ExtraTreesClassifier Acuraccy: ',etc_score)



#4) RESULTS

labels = ['SVM','NBM','LR','AdaBoost','RF','GBC','ETC']
all_scores = [svm_score,nbm_score,lg_score,adaBoost_score,rf_score,GrB_score,etc_score]

x = numpy.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot()
rects1 = ax.bar(x, all_scores, width)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of the experimented Classifiers for KC2 dataset')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim(0.70, 0.99)
# ax.legend(loc="upper center", bbox_to_anchor=(0.4, 1.20))

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
    
        ax.annotate('{}'.format(round(100*height,3)),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
               
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
plt.show()
result_dir = '/content/drive/MyDrive/softwareBugPred/bugPridiction/results'
plt.savefig(f"{result_dir}/test.png")





# labels = ['SVM','KNN','NBM','AdaBoost','RF']
# allTs = [0.89999,0.89333,0.8633,0.8744,0.8844]
# allT = [0.7444,0.8744,0.80,0.7944,0.84444]

# x = numpy.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig = plt.figure(figsize=(12,7))
# ax = fig.add_subplot()
# rects1 = ax.bar(x - width/2, allTs, width, label='Dataset containing social media features and addition to textual features')
# rects2 = ax.bar(x + width/2, allT, width, label='Dataset containing only textual features')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy of the experimented Classifiers')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# plt.ylim(0.70, 0.92)
# ax.legend(loc="upper center", bbox_to_anchor=(0.4, 1.20))

# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
    
#         ax.annotate('{}'.format(round(100*height,3)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
               
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# autolabel(rects1)
# autolabel(rects2)
# plt.show()


