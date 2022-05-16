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


#no mistakes
clf = RandomForestClassifier(n_estimators = 1000, max_depth=2, random_state=0)
clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx] , testlabel[train_idx])
####Split Norm
y_preds = clf.predict_proba(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])[:,1]
fpr_no_mistakes, tpr_no_mistakes, _ = roc_curve(testlabel[test_idx], y_preds)

#mistake 1: feature norm using full dataset
clf = RandomForestClassifier(n_estimators = 1000, max_depth=2, random_state=0)
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_training_idx],testlabel[train_idx])
fpr_mistakes1, tpr_mistakes1, _ = roc_curve(testlabel2, clf.predict_proba(institution2_norm[:,univariate_diff_features_training_idx])[:,1])

#mistake 2: feature selection using full dataset
clf = RandomForestClassifier(n_estimators = 1000, max_depth=2, random_state=0)
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
fpr_mistakes12, tpr_mistakes12, _ = roc_curve(testlabel2, clf.predict_proba(institution2_norm[:,univariate_diff_features_idx])[:,1])

#mistake 3: model selection using test set - relies on prior knowledge from Table 1
gnb = GaussianNB()
gnb.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
fpr_mistakes123, tpr_mistakes123, _ = roc_curve(testlabel2, gnb.predict_proba(institution2_norm[:,univariate_diff_features_idx])[:,1])

#mistake 4: no external test set
gnb = GaussianNB()
gnb.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
y_preds = gnb.predict_proba(testset_smote_norm[test_idx][:,univariate_diff_features_idx])[:,1]
fpr_mistakes1234, tpr_mistakes1234, _ = roc_curve(testlabel[test_idx], y_preds) 

#mistake 5: hyperparameter selection using full dataset
clf = svm.SVC(C=1, kernel='sigmoid', coef0=0.01)
clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx],testlabel[train_idx])
y_preds = clf.decision_function(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
fpr_mistakes12345, tpr_mistakes12345, _ = roc_curve(testlabel[test_idx], y_preds)

#mistake 6: reporting results on full dataset
clf = RandomForestClassifier(n_estimators = 1000, max_depth=4, random_state=0)
clf.fit(testset_smote_norm[:,univariate_diff_features_idx],testlabel)
fpr_mistakes123456, tpr_mistakes123456, _ = roc_curve(testlabel, clf.predict_proba(testset_smote_norm[:,univariate_diff_features_idx])[:,1])

#plot ROCs
plt.figure(figsize=(5,5))
plt.plot(fpr_no_mistakes, fpr_no_mistakes, linestyle = '--', label = 'Random chance')
plt.plot(fpr_no_mistakes, tpr_no_mistakes, linestyle = '-', label = 'No Mistakes')
plt.plot(fpr_mistakes1, tpr_mistakes1, linestyle = '-', label = 'Mistake 1')
plt.plot(fpr_mistakes12, tpr_mistakes12, linestyle = '-', label = 'Mistakes 1, 2')
plt.plot(fpr_mistakes123, tpr_mistakes123, linestyle = '-', label = 'Mistakes 1, 2, 3')
plt.plot(fpr_mistakes1234, tpr_mistakes1234, linestyle = '-', label = 'Mistakes 1, 2, 3, 4')
plt.plot(fpr_mistakes12345, tpr_mistakes12345, linestyle = '-', label = 'Mistakes 1, 2, 3, 4, 5')
plt.plot(fpr_mistakes123456, tpr_mistakes123456, linestyle = '-', label = 'Mistakes 1, 2, 3, 4, 5, 6')
plt.legend(bbox_to_anchor = (1,1))
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()