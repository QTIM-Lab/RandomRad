import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def dataset_norm(x):
	return (x - np.mean(x))/np.std(x)
def train_split_norm(y,wrt):
	if np.std(wrt) == 0:
		result = (y - np.mean(wrt))/(np.std(wrt) + 0.0001)
	else:
		result = (y - np.mean(wrt))/np.std(wrt)
	return result

testset_smote = np.load('randomrad_smote.npy')
testlabel = np.load('randomrad_label.npy')

testset_smote_norm = testset_smote.copy()
for column in range(testset_smote_norm.shape[1]):
	testset_smote_norm[:,column] = dataset_norm(testset_smote[:,column])

assn_no_partition = np.zeros(10)
assn_train_norm = np.zeros(10)
assn_split_norm = np.zeros(10)
diff_no_partition = np.zeros(10)
diff_train_norm = np.zeros(10)
diff_split_norm = np.zeros(10)

for q in np.arange(10):
	print(q)
	train_idx, test_idx = train_test_split(np.arange(len(testlabel)), stratify= testlabel, test_size = 0.3)

	testset_smote_norm_correct_train = testset_smote[train_idx].copy()
	testset_smote_norm_correct_test = testset_smote[test_idx].copy()
	testset_smote_norm_correct_test_sep = testset_smote[test_idx].copy()

	for column in range(testset_smote_norm_correct_train.shape[1]):
		testset_smote_norm_correct_train[:,column] = dataset_norm(testset_smote[train_idx,column])
		testset_smote_norm_correct_test[:,column] = train_split_norm(testset_smote[test_idx,column], testset_smote[train_idx,column]) #norm strategy 1
		testset_smote_norm_correct_test_sep[:,column] = dataset_norm(testset_smote[test_idx,column]) #norm strategy 2
	
	testset_smote_norm_correct_train = np.nan_to_num(testset_smote_norm_correct_train, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
	testset_smote_norm_correct_test = np.nan_to_num(testset_smote_norm_correct_test, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
	testset_smote_norm_correct_test_sep = np.nan_to_num(testset_smote_norm_correct_test_sep, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
	
	clf = LassoCV(cv = 3)
	clf.fit(testset_smote_norm, testlabel)
	univariate_assn_features_idx= np.where(abs(clf.coef_)>0)[0]

	clf = LassoCV(cv = 3)
	clf.fit(testset_smote_norm_correct_train, testlabel[train_idx])
	univariate_assn_features_training_idx= np.where(abs(clf.coef_)>0)[0]
	if len(univariate_assn_features_training_idx)==0:
		univariate_assn_features_training_idx = np.arange(testset_smote_norm_correct_train.shape[1])

	univariate_diff_features_idx = []
	univariate_diff_features_training_idx = []
	for i in range(testset_smote_norm.shape[1]):
		if mannwhitneyu(testset_smote_norm[np.where(testlabel==0)[0],i],
					testset_smote_norm[np.where(testlabel==1)[0],i]).pvalue < 0.05:
			univariate_diff_features_idx.append(i)
		if mannwhitneyu(testset_smote_norm_correct_train[np.where(testlabel[train_idx]==0)[0],i],
					testset_smote_norm_correct_train[np.where(testlabel[train_idx]==1)[0],i]).pvalue < 0.05:
			univariate_diff_features_training_idx.append(i)
		
	#predictive value of features that we already know to be associated with outcome
	clf = svm.SVC()
	clf.fit(testset_smote_norm[train_idx][:,univariate_assn_features_idx], testlabel[train_idx])
	y_preds = clf.predict(testset_smote_norm[test_idx][:,univariate_assn_features_idx])
	assn_leakage_score = round(accuracy_score(testlabel[test_idx], y_preds),3)
	
	#done with train/test split throughout (Globally)
	clf = svm.SVC()
	clf.fit(testset_smote_norm_correct_train[:,univariate_assn_features_training_idx], testlabel[train_idx])
	y_preds = clf.predict(testset_smote_norm_correct_test[:,univariate_assn_features_training_idx])
	assn_no_leakage_score = round(accuracy_score(testlabel[test_idx], y_preds),3)
	
	y_preds = clf.predict(testset_smote_norm_correct_test_sep[:,univariate_assn_features_training_idx])
	assn_no_leakage_score_sep = round(accuracy_score(testlabel[test_idx], y_preds),3)
	
	#predictive value of features we already know are significantly split by outcome
	clf = svm.SVC()
	clf.fit(testset_smote_norm[train_idx][:,univariate_diff_features_idx], testlabel[train_idx])
	y_preds = clf.predict(testset_smote_norm[test_idx][:,univariate_diff_features_idx])
	diff_leakage_score = round(accuracy_score(testlabel[test_idx], y_preds),3)
	
	#done with train/test split throughout (Globally)
	clf = svm.SVC()
	clf.fit(testset_smote_norm_correct_train[:,univariate_diff_features_training_idx], testlabel[train_idx])
	y_preds = clf.predict(testset_smote_norm_correct_test[:,univariate_diff_features_training_idx])
	diff_no_leakage_score = round(accuracy_score(testlabel[test_idx], y_preds),3)
	
	y_preds = clf.predict(testset_smote_norm_correct_test_sep[:,univariate_diff_features_training_idx])
	diff_no_leakage_score_sep = round(accuracy_score(testlabel[test_idx], y_preds),3)

	assn_no_partition[q] = assn_leakage_score
	assn_train_norm[q] = assn_no_leakage_score
	assn_split_norm[q] = assn_no_leakage_score_sep
	diff_no_partition[q] = diff_leakage_score
	diff_train_norm[q] = diff_no_leakage_score
	diff_split_norm[q] = diff_no_leakage_score_sep
	
fig, ax = plt.subplots(figsize=(10,5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.plot([0,1],[np.mean(assn_no_partition),np.mean(assn_train_norm)],'ro-', label = 'LassoCV') 
plt.plot([0,1],[np.mean(diff_no_partition),np.mean(diff_train_norm)],color ='orange', linestyle='-', marker='o', label = 'MannWhitney') 
plt.annotate(str(round(np.mean(assn_no_partition),3)),[-.1,np.mean(assn_no_partition)])
plt.annotate(str(round(np.mean(assn_train_norm),3)),[1.03,np.mean(assn_train_norm)])
plt.annotate(str(round(np.mean(diff_no_partition),3)),[-.1,np.mean(diff_no_partition)])
plt.annotate(str(round(np.mean(diff_train_norm),3)),[1.03,np.mean(diff_train_norm)])
plt.grid(b=None)
plt.yticks([])
plt.xlim(-.2,1.3)
plt.xticks([0,1],labels=['Inconsistent Partitioning','Consistent Partitioning'])
lgd = plt.legend(bbox_to_anchor = (1.2,1))
plt.show()

#np.save('assn_no_partition.npy',assn_no_partition)
#np.save('assn_train_norm.npy',assn_train_norm)
#np.save('assn_split_norm.npy',assn_split_norm)
#np.save('diff_no_partition.npy',diff_no_partition)
#np.save('diff_train_norm.npy',diff_train_norm)
#np.save('diff_split_norm.npy',diff_split_norm)