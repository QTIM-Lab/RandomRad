import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

testset_smote = np.load('randomrad_smote.npy')
testlabel = np.load('randomrad_label.npy')
institution2 = np.load('randomrad_smote_Institution2.npy')
testlabel2 = np.load('randomrad_label_Institution2.npy')
def dataset_norm(x):
    return (x - np.mean(x))/np.std(x)
def train_split_norm(x,y):
    return (x - np.mean(y))/np.std(y)

testset_smote_norm = testset_smote.copy()
#we will now normalize all the data together to demonstrate the unfair advatage that confers on further experimention 
for column in range(testset_smote_norm.shape[1]):
    testset_smote_norm[:,column] = dataset_norm(testset_smote[:,column])

train_idx, test_idx = train_test_split(np.arange(len(testlabel)), stratify= testlabel, test_size = 0.3)
###the first normalization strategy will be to normalize the test set using the mean and standard deviation of the training set
testset_smote_norm_correct_train = testset_smote[train_idx].copy()
testset_smote_norm_correct_test = testset_smote[test_idx].copy()

###the second normalization strategy will be to normalize the test set using its own mean and standard deviation 
testset_smote_norm_correct_test_sep = testset_smote[test_idx].copy()
for column in range(testset_smote_norm_correct_train.shape[1]):
    testset_smote_norm_correct_train[:,column] = dataset_norm(testset_smote[train_idx,column])
    testset_smote_norm_correct_test[:,column] = train_split_norm(testset_smote[test_idx,column], testset_smote[train_idx,column]) #norm strategy 1
    testset_smote_norm_correct_test_sep[:,column] = dataset_norm(testset_smote[test_idx,column]) #norm strategy 2


institution2_norm = institution2.copy()
for column in range(institution2_norm.shape[1]):
    institution2_norm[:,column] = dataset_norm(institution2[:,column])


clf = LassoCV(cv = 3)
clf.fit(testset_smote_norm, testlabel)
univariate_assn_features_idx= np.where(abs(clf.coef_)>0)[0]

clf = LassoCV(cv = 3)
clf.fit(testset_smote_norm_correct_train, testlabel[train_idx])
univariate_assn_features_training_idx= np.where(abs(clf.coef_)>0)[0]


univariate_diff_features_idx = []
univariate_diff_features_training_idx = []
for i in range(testset_smote_norm.shape[1]):
    if mannwhitneyu(testset_smote_norm[np.where(testlabel==0)[0],i],
                testset_smote_norm[np.where(testlabel==1)[0],i]).pvalue < 0.05:
        univariate_diff_features_idx.append(i)
    if mannwhitneyu(testset_smote_norm_correct_train[np.where(testlabel[train_idx]==0)[0],i],
                testset_smote_norm_correct_train[np.where(testlabel[train_idx]==1)[0],i]).pvalue < 0.05:
        univariate_diff_features_training_idx.append(i)


methods = []
test_roc = []
train_roc_norm_correct = []
test_roc_norm_train = []
test_roc_norm_split = []

clf = RandomForestClassifier(n_estimators = 1000, max_depth=2, random_state=0)
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx]) 
methods.append('Random Forest')
#test accuracy
y_preds = clf.predict_proba(testset_smote_norm[test_idx][:,univariate_diff_features_idx])[:,1]
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))

clf = RandomForestClassifier(n_estimators = 1000, max_depth=2, random_state=0)
clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx] , testlabel[train_idx])
y_preds = clf.predict_proba(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])[:,1]
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx],y_preds))
####Train Norm
y_preds = clf.predict_proba(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])[:,1]
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))
####Split Norm
y_preds = clf.predict_proba(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])[:,1]
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))



clf = svm.SVC(C=1.0, kernel='rbf')
methods.append('Support Vector Machine: rbf')
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
#test accuracy
y_preds = clf.decision_function(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))

clf = svm.SVC(C=1.0, kernel='rbf')
clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx],testlabel[train_idx])
y_preds = clf.decision_function(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx], y_preds))
####Train Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))
####Split Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))



clf = svm.SVC(C=1.0, kernel='linear')
methods.append('Support Vector Machine: linear')
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
#test accuracy
y_preds = clf.decision_function(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))

clf = svm.SVC(C=1.0, kernel='linear')
clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx],testlabel[train_idx])
y_preds = clf.decision_function(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx], y_preds))
####Train Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))
####Split Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))



clf = svm.SVC(C=1.0, kernel='poly', degree = 3)
methods.append('Support Vector Machine: polynomial')
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
#test accuracy
y_preds = clf.decision_function(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))

clf = svm.SVC(C=1.0, kernel='poly', degree = 3)
clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx],testlabel[train_idx])
y_preds = clf.decision_function(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx], y_preds))
####Train Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))
fpr_test_poly3_TN, tpr_test_poly3_TN, _ = roc_curve(testlabel[test_idx], y_preds) #1-19-21
####Split Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))



clf = svm.SVC(C=1.0, kernel='sigmoid')
methods.append('Support Vector Machine: sigmoid')
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
#test accuracy
y_preds = clf.decision_function(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))

clf = svm.SVC(C=1.0, kernel='sigmoid')
clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx],testlabel[train_idx])
y_preds = clf.decision_function(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx], y_preds))
####Train Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))
####split Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))



clf = svm.SVC(C=1.0, kernel='poly', degree = 4)
methods.append('Support Vector Machine: 4th degree polynomial')
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
#test accuracy
y_preds = clf.decision_function(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))

clf = svm.SVC(C=1.0, kernel='poly', degree = 4)
clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx],testlabel[train_idx])
y_preds = clf.decision_function(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx], y_preds))
####Train Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))
####Split Norm
y_preds = clf.decision_function(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))



methods.append('Gaussian Naive Bayes')
gnb = GaussianNB()
gnb.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
y_preds = gnb.predict_proba(testset_smote_norm[test_idx][:,univariate_diff_features_idx])[:,1]
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))

gnb = GaussianNB()
gnb.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx],testlabel[train_idx])
y_preds = gnb.predict_proba(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])[:,1]
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx], y_preds))
####Train Norm
y_preds = gnb.predict_proba(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])[:,1]
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))
####Split Norm
y_preds = gnb.predict_proba(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])[:,1]
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))



K.clear_session()
feature_input = Input((len(univariate_diff_features_idx),))
x = Dense(16, activation = 'relu')(feature_input)
class_output = Dense(1, activation = 'relu')(x)
BasicModel = Model(feature_input,class_output)
methods.append('Shallow Neural Network')
BasicModel.compile(loss = 'mean_squared_error', metrics= ['accuracy'])

problematic_history = BasicModel.fit(x = testset_smote_norm[:,univariate_diff_features_idx], y = testlabel, epochs = 100, verbose = 0)
####Dataset Norm
y_preds = BasicModel.predict(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
test_roc.append(roc_auc_score(testlabel[test_idx], y_preds))


K.clear_session()
feature_input = Input((len(univariate_diff_features_training_idx),))
x = Dense(16, activation = 'relu')(feature_input)
class_output = Dense(1, activation = 'relu')(x)
BasicModel = Model(feature_input,class_output)
BasicModel.compile(loss = 'mean_squared_error', metrics= ['accuracy'])

basic_history = BasicModel.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx], testlabel[train_idx], epochs = 100, verbose = 0)
y_preds = BasicModel.predict(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx])
train_roc_norm_correct.append(roc_auc_score(testlabel[train_idx], y_preds))

####Train Norm
y_preds = BasicModel.predict(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])
test_roc_norm_train.append(roc_auc_score(testlabel[test_idx], y_preds))

####Split Norm
y_preds = BasicModel.predict(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])
test_roc_norm_split.append(roc_auc_score(testlabel[test_idx], y_preds))


ML_shotgun_df = pd.DataFrame(columns=methods[0:8], data= [test_roc, train_roc_norm_correct, test_roc_norm_train, test_roc_norm_split], 
                             index = ['Dataset Norm Test AUC','Sep Norm Train AUC','Train Norm Test AUC','Split Norm Test AUC']).T

ML_shotgun_df.index.name = 'ML Method'
print(ML_shotgun_df)