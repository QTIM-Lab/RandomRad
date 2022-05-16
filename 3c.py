import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import svm
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
#we will now normalize all the data together to demonstrate the unfair advatage that confers on further experimention 
for column in range(testset_smote_norm.shape[1]):
    testset_smote_norm[:,column] = dataset_norm(testset_smote[:,column])
    
force_split_auc = []
for i in range(100):
    train_idx, test_idx = train_test_split(np.arange(len(testlabel)), stratify= testlabel, test_size = 0.3)
    
    testset_smote_norm_correct_train = testset_smote[train_idx].copy()
    testset_smote_norm_correct_test = testset_smote[test_idx].copy()
    testset_smote_norm_correct_test_sep = testset_smote[test_idx].copy()

    for column in range(testset_smote_norm_correct_train.shape[1]):
        testset_smote_norm_correct_train[:,column] = dataset_norm(testset_smote[train_idx,column])
        testset_smote_norm_correct_test[:,column] = train_split_norm(testset_smote[test_idx,column], testset_smote[train_idx,column]) #norm strategy 1
        testset_smote_norm_correct_test_sep[:,column] = dataset_norm(testset_smote[test_idx,column]) #norm strategy 2
    
    clf = svm.SVC() #in our manuscript we populated with optimized hyperparameters
    clf.fit(testset_smote_norm_correct_train,testlabel[train_idx])
    y_preds = clf.decision_function(testset_smote_norm_correct_test)
    force_split_auc.append(roc_auc_score(testlabel[test_idx], y_preds))

plt.hist(force_split_auc,color ='k')
plt.xticks(np.arange(0,1,.1))
plt.xlabel('ROC-AUC Score')
plt.show()