import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
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

dataset_norm_scores = []
train_norm_scores = []
split_norm_scores = []
for i in np.arange(100):
    #Split the Data
    train_idx, test_idx = train_test_split(np.arange(len(testlabel)), stratify= testlabel, test_size = 0.3)
    
    reg = LassoCV(cv = 10, random_state = 0).fit(testset_smote_norm, testlabel)
    dataset_norm_scores.append(reg.score(testset_smote_norm[test_idx], testlabel[test_idx]))
    
    #Normalize by all strategies
    ###the first normalization strategy will be to normalize the test set using the mean and standard deviation of the training set
    testset_smote_norm_correct_train = testset_smote[train_idx].copy()
    testset_smote_norm_correct_test = testset_smote[test_idx].copy()
    
    ###the second normalization strategy will be to normalize the test set using its own mean and standard deviation 
    testset_smote_norm_correct_test_sep = testset_smote[test_idx].copy()
    
    for column in range(testset_smote_norm_correct_train.shape[1]):
        testset_smote_norm_correct_train[:,column] = np.nan_to_num(dataset_norm(testset_smote[train_idx,column]))
        testset_smote_norm_correct_test[:,column] = np.nan_to_num(train_split_norm(testset_smote[test_idx,column], testset_smote[train_idx,column])) #norm strategy 1
        testset_smote_norm_correct_test_sep[:,column] = np.nan_to_num(dataset_norm(testset_smote[test_idx,column])) #norm strategy 2
        
    #Get model scores when trained with partitioning
    reg = LassoCV(cv = 10, random_state = 0).fit(testset_smote_norm_correct_train, testlabel[train_idx])
    train_norm_scores.append(reg.score(testset_smote_norm_correct_test, testlabel[test_idx]))
    split_norm_scores.append(reg.score(testset_smote_norm_correct_test_sep, testlabel[test_idx]))

fig, ax = plt.subplots()
plt.arrow(np.mean(dataset_norm_scores),0,np.mean(train_norm_scores)-np.mean(dataset_norm_scores),0, width = 0.0001, head_width = 0.3, head_length = 0.01, color = 'k')
x = np.mean(np.array(train_norm_scores)) 
y = np.std(np.array(train_norm_scores))
plt.arrow(np.mean(dataset_norm_scores),1,np.mean(split_norm_scores)-np.mean(dataset_norm_scores),0, width = 0.0001, head_width = 0.3, head_length = 0.01, color = 'k')
x = np.mean(np.array(split_norm_scores)) 
y = np.std(np.array(split_norm_scores))
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks(np.arange(2),labels= ['Train Norm','Split Norm'])
plt.axvline(0,linestyle='--',color='r', linewidth=1)
plt.xlabel('Change in LassoCV $R^2$ for Test Set (CV=10)', font = 'serif', fontsize = 12)
plt.show()