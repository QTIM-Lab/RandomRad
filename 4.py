import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from MRMR import mrmr


testset_smote = np.load('randomrad_smote.npy')
testset_smote_HNSCC = np.load('randomrad_smote_HNSCC.npy')

def dataset_norm(x):
    return (x - np.mean(x))/np.std(x)
def train_split_norm(x,y):
    return (x - np.mean(y))/np.std(y)
    
testset_smote_norm = testset_smote.copy()
testset_smote_HNSCC_norm = testset_smote_HNSCC.copy()
#we will now normalize all the data together to demonstrate the unfair advatage that confers on further experimention 
for column in range(testset_smote_norm.shape[1]):
    testset_smote_norm[:,column] = dataset_norm(testset_smote[:,column])
for column in range(testset_smote_HNSCC_norm.shape[1]):
    testset_smote_HNSCC_norm[:,column] = dataset_norm(testset_smote_HNSCC[:,column])

clinical_vars = pd.read_csv('ClinicalData/TCGA_LGG_clinical_data.csv', index_col = 1) #TCGA Clinical Data downloaded from CBioPortal
clinical_vars_HNSCC = pd.read_csv('ClinicalData/TCGA_HNSCC_clinical_data.csv', index_col = 0) 
clinical_vars = clinical_vars.loc[~clinical_vars.index.duplicated(keep='first')]
clinical_vars_HNSCC = clinical_vars_HNSCC.loc[~clinical_vars_HNSCC.index.duplicated(keep='first')]
LGG_annotated_cases = np.load('ClinicalData/LGG_annotated_cases.npy', allow_pickle = True)
HNSCC_annotated_cases = np.load('ClinicalData/HNSCC_annotated_cases.npy', allow_pickle = True)

OS = np.array(clinical_vars.loc[LGG_annotated_cases,'Overall Survival (Months)'])
OS_HNSCC = np.array(clinical_vars_HNSCC.loc[HNSCC_annotated_cases,'Overall Survival (Months)'])
OS_severity = OS>=np.median(OS)
OS_severity = OS_severity*1
OS_severity_HNSCC = OS_HNSCC>=np.median(OS_HNSCC)
OS_severity_HNSCC = OS_severity_HNSCC*1

LGG_fts_assn_OS = []
for i in np.arange(testset_smote.shape[1]):
    if spearmanr(testset_smote_norm[:,i],OS).pvalue < 0.3 and mannwhitneyu(testset_smote_norm[np.where(OS_severity==0)[0]][:,i],testset_smote_norm[np.where(OS_severity==1)[0]][:,i]).pvalue<0.3:
        LGG_fts_assn_OS.append(i)

#model = KMeans()
#visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette' ,timings= True) #testing 2-10 clusters
#visualizer.fit(testset_smote_norm[:,LGG_fts_assn_OS])   

model = KMeans(n_clusters = 2) #visualizer.elbow_value_
model.fit(testset_smote_norm[:,LGG_fts_assn_OS])
cluster_labels = model.labels_

heatcorr = np.zeros((testset_smote_norm.shape[0], testset_smote_norm.shape[0]))
count_x = 0
for i in range(testset_smote_norm.shape[0]):
    count_y = 0 
    count_x = count_x + 1
    for j in range(testset_smote_norm.shape[0]):
        heatcorr[count_x -1, count_y] = spearmanr(testset_smote_norm[np.argsort(cluster_labels)][i,LGG_fts_assn_OS], 
                                                 testset_smote_norm[np.argsort(cluster_labels)][j,LGG_fts_assn_OS])[0]
        count_y = count_y + 1

HNSCC_fts_assn_OS = []
for i in np.arange(testset_smote_HNSCC.shape[1]):
    if spearmanr(testset_smote_HNSCC_norm[:,i],OS_HNSCC).pvalue < 0.04 and mannwhitneyu(testset_smote_HNSCC_norm[np.where(OS_severity_HNSCC==0)[0]][:,i],testset_smote_HNSCC_norm[np.where(OS_severity_HNSCC==1)[0]][:,i]).pvalue<0.04:
        HNSCC_fts_assn_OS.append(i)

model = KMeans(n_clusters = 2) #visualizer.elbow_value_
model.fit(testset_smote_HNSCC_norm[:,HNSCC_fts_assn_OS])
cluster_labels_HNSCC = model.labels_

heatcorr_HNSCC = np.zeros((testset_smote_HNSCC_norm.shape[0], testset_smote_HNSCC_norm.shape[0]))
count_x = 0
for i in range(testset_smote_HNSCC_norm.shape[0]):
    count_y = 0 
    count_x = count_x + 1
    for j in range(testset_smote_HNSCC_norm.shape[0]):
        heatcorr_HNSCC[count_x -1, count_y] = spearmanr(testset_smote_HNSCC_norm[np.argsort(cluster_labels_HNSCC)][i,HNSCC_fts_assn_OS], 
                                                 testset_smote_HNSCC_norm[np.argsort(cluster_labels_HNSCC)][j,HNSCC_fts_assn_OS])[0]
        count_y = count_y + 1


plt.figure(figsize=(8,7))
ax = sns.heatmap(heatcorr, cmap = 'Blues', cbar=True)
plt.xlabel('Cases', font = 'serif', fontsize = 14)
plt.ylabel('Cases', font = 'serif', fontsize = 14)
plt.xticks([], [])
plt.yticks([], [])
plt.title('LGG Case Correlations after Consensus Clustering', font = 'serif', fontsize = 14)
cbar = ax.collections[0].colorbar
cbar.set_label(label='Correlation',fontsize = 14, fontfamily = 'serif')
cbar.ax.tick_params(labelsize=12)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('serif')
plt.show()


plt.figure(figsize=(8,1))
ax = sns.heatmap(OS[np.argsort(cluster_labels)].reshape(1,65), cmap = 'bone_r', cbar=True, cbar_kws = {'ticks':[0,1],'aspect':5.5})
plt.xlabel('Cases', font = 'serif', fontsize = 14)
plt.ylabel('OS', font = 'serif', fontsize = 14)
plt.xticks([], [])
plt.yticks([], [])
cbar = ax.collections[0].colorbar
cbar.set_label(label='Prognosis Severity',fontsize = 14, fontfamily = 'serif')
cbar.ax.tick_params(labelsize=0)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('serif')
plt.show()


fig, ax = plt.subplots(1,figsize=(6.7,1), frameon=False)
plt.barh('Cluster',len(np.where(cluster_labels==0)[0]), color = 'pink')
plt.barh('Cluster',len(np.where(cluster_labels==1)[0]), left = len(np.where(cluster_labels==0)[0]), color = 'Navy') #84e8f5')
plt.xlabel('Cases', font = 'serif', fontsize = 14)
plt.ylabel('Cluster', font = 'serif', fontsize = 14)
plt.xticks(np.arange(0,66,5), font = 'serif', fontsize = 11)
plt.yticks([], [])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.text(len(np.where(cluster_labels==0)[0])/2,-.07,str(len(np.where(cluster_labels==0)[0])), font = 'serif', fontsize =12)
plt.text(len(np.where(cluster_labels==0)[0])+(len(np.where(cluster_labels==1)[0])/2)-1,-.07,str(len(np.where(cluster_labels==1)[0])), font = 'serif', fontsize =13, color = 'white')
plt.show()      


print('Number of cases/controls in Cluster 0: ',sum(OS_severity[np.where(cluster_labels==0)]),'/', len(OS_severity[np.where(cluster_labels==0)]), 
      '[',round(sum(OS_severity[np.where(cluster_labels==0)])/len(OS_severity[np.where(cluster_labels==0)]),3)*100,'%]')
print('Number of cases/controls in Cluster 1: ',sum(OS_severity[np.where(cluster_labels==1)]), '/', len(OS_severity[np.where(cluster_labels==1)]) ,
      '[',round(sum(OS_severity[np.where(cluster_labels==1)])/len(OS_severity[np.where(cluster_labels==1)]),3)*100,'%]')


plt.figure(figsize=(8,7))
ax = sns.heatmap(heatcorr_HNSCC, cmap = 'Greens', cbar=True)
plt.xlabel('Cases', font = 'serif', fontsize = 14)
plt.ylabel('Cases', font = 'serif', fontsize = 14)
plt.xticks([], [])
plt.yticks([], [])
plt.title('HNSCC Case Correlations after Consensus Clustering', font = 'serif', fontsize = 14)
cbar = ax.collections[0].colorbar
cbar.set_label(label='Correlation',fontsize = 14, fontfamily = 'serif')
cbar.ax.tick_params(labelsize=12)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('serif')
plt.show()


plt.figure(figsize=(8,1))
ax = sns.heatmap(OS_HNSCC[np.argsort(cluster_labels_HNSCC)].reshape(1,115), cmap = 'bone_r', cbar=True, cbar_kws = {'ticks':[0,1],'aspect':5.5})
plt.xlabel('Cases', font = 'serif', fontsize = 14)
plt.ylabel('OS', font = 'serif', fontsize = 14)
plt.xticks([], [])
plt.yticks([], [])
cbar = ax.collections[0].colorbar
cbar.set_label(label='Prognosis Severity',fontsize = 14, fontfamily = 'serif')
cbar.ax.tick_params(labelsize=0)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('serif')
plt.show()


fig, ax = plt.subplots(1,figsize=(6.7,1), frameon=False)
plt.barh('Cluster',len(np.where(cluster_labels_HNSCC==0)[0]), color = 'pink')
plt.barh('Cluster',len(np.where(cluster_labels_HNSCC==1)[0]), left = len(np.where(cluster_labels_HNSCC==0)[0]), color = 'Navy') #84e8f5')
plt.xlabel('Cases', font = 'serif', fontsize = 14)
plt.ylabel('Cluster', font = 'serif', fontsize = 14)
plt.xticks(np.arange(0,116,25), font = 'serif', fontsize = 11)
plt.yticks([], [])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.text(len(np.where(cluster_labels_HNSCC==0)[0])/2,-.07,str(len(np.where(cluster_labels_HNSCC==0)[0])), font = 'serif', fontsize =12)
plt.text(len(np.where(cluster_labels_HNSCC==0)[0])+(len(np.where(cluster_labels_HNSCC==1)[0])/2)-1,-.07,str(len(np.where(cluster_labels==1)[0])), font = 'serif', fontsize =13, color = 'white')
plt.show()      


print('Number of cases/controls in Cluster 0: ',sum(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==0)]),'/', len(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==0)]), 
      '[',round(sum(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==0)])/len(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==0)]),3)*100,'%]')
print('Number of cases/controls in Cluster 1: ',sum(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==1)]), '/', len(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==1)]) ,
      '[',round(sum(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==1)])/len(OS_severity_HNSCC[np.where(cluster_labels_HNSCC==1)]),3)*100,'%]')