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
    
multi_site_trained_on = []
multi_site_tested_on = []
multi_site_roc_auc = []

multi_site_trained_on.append('Positives from I1 and Negatives from I2')

clf = svm.SVC(C=1, kernel='sigmoid', gamma = 'scale', coef0 = 0, tol = .0001, random_state = 1)
clf.fit(np.concatenate([testset_smote_norm[np.where(testlabel == 1)[0]],institution2_norm[np.where(testlabel2 == 0)[0]]], axis = 0),
        np.concatenate([testlabel[np.where(testlabel == 1)[0]],testlabel[np.where(testlabel2 == 0)[0]]]))  

multi_site_tested_on.append('Negatives from I1 and Positives from I2')
y_preds = clf.decision_function(np.concatenate([testset_smote_norm[np.where(testlabel == 0)[0]], institution2_norm[np.where(testlabel2 == 1)[0]]], axis = 0))
multi_site_roc_auc.append(round(roc_auc_score(np.concatenate([testlabel[np.where(testlabel == 0)[0]], testlabel2[np.where(testlabel2 == 1)[0]]]), y_preds),3))

multi_site_trained_on.append('Positives from I2 and Negatives from I1')

clf = svm.SVC(C=1, kernel='sigmoid', gamma = 'scale', coef0 = 0, tol = .0001, random_state = 1)
clf.fit(np.concatenate([testset_smote_norm[np.where(testlabel == 0)[0]],institution2_norm[np.where(testlabel2 == 1)[0]]], axis = 0),
        np.concatenate([testlabel[np.where(testlabel == 0)[0]],testlabel[np.where(testlabel2 == 1)[0]]]))  

multi_site_tested_on.append('Negatives from I2 and Positives from I1')
y_preds = clf.decision_function(np.concatenate([testset_smote_norm[np.where(testlabel == 1)[0]], institution2_norm[np.where(testlabel2 == 0)[0]]], axis = 0))
multi_site_roc_auc.append(round(roc_auc_score(np.concatenate([testlabel[np.where(testlabel == 1)[0]], testlabel2[np.where(testlabel2 == 0)[0]]]), y_preds),3))

multi_site_trained_on.append('I1')

clf = svm.SVC(C=1, kernel='sigmoid', gamma = 'scale', coef0 = 0, tol = .0001, random_state = 1)
clf.fit(testset_smote_norm, testlabel)  

multi_site_tested_on.append('I2')
y_preds = clf.decision_function(institution2_norm)
multi_site_roc_auc.append(round(roc_auc_score(testlabel2, y_preds),3))

multi_site_trained_on.append('I2')

clf = svm.SVC(C=1, kernel='sigmoid', gamma = 'scale', coef0 = 0, tol = .0001, random_state = 1)
clf.fit(institution2_norm, testlabel2)  

multi_site_tested_on.append('I1')
y_preds = clf.decision_function(testset_smote_norm)
multi_site_roc_auc.append(round(roc_auc_score(testlabel, y_preds),3))

train_idx, test_idx = train_test_split(np.arange(len(testlabel)), stratify= testlabel)
train_idx2, test_idx2 = train_test_split(np.arange(len(testlabel2)), stratify= testlabel2)

multi_site_trained_on.append('Training sets of I1 and I2')

clf = svm.SVC(C=1, kernel='sigmoid', gamma = 'scale', coef0 = 0, tol = .0001, random_state = 1)
clf.fit(np.concatenate([testset_smote_norm[train_idx],institution2_norm[train_idx2]], axis = 0),
        np.concatenate([testlabel[train_idx],testlabel[train_idx2]]))  

multi_site_tested_on.append('Test sets of I1 and I2')
y_preds = clf.decision_function(np.concatenate([testset_smote_norm[test_idx], institution2_norm[test_idx2]], axis = 0))
multi_site_roc_auc.append(round(roc_auc_score(np.concatenate([testlabel[test_idx], testlabel2[test_idx2]]), y_preds),3))

I1I2_df = pd.DataFrame(index = ['Training Set','Testing Set','Test Set AUC'], 
             data = [multi_site_trained_on, multi_site_tested_on, multi_site_roc_auc]).T

print(I1I2_df)