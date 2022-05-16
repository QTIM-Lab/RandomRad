import os
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, bartlett
from sklearn.metrics import mutual_info_score as MIS

def mrmr_discretize(x):
    egbins = np.histogram_bin_edges(x, bins = 256)
    return np.digitize(x, egbins)


def mrmr(data, labels, num_of_features):
    final_features_idx = []
    first_feature_idx = np.where([bartlett(data.iloc[:,i], labels).statistic for i in np.arange(data.shape[1])] == np.max([bartlett(data.iloc[:,i], labels).statistic for i in np.arange(data.shape[1])]))[0][0]
    final_features_idx.append(first_feature_idx)
    chop_data = data.drop(data.columns[first_feature_idx], axis = 1).copy()
    D = MIS(data.iloc[:,first_feature_idx]>np.mean(data.iloc[:,first_feature_idx]), labels)
    S = 0
    for g in np.arange(num_of_features -1):
        D_sum = np.sum([MIS(mrmr_discretize(data.iloc[:,p]), labels) for p in final_features_idx])
        S_sum = np.sum([MIS(mrmr_discretize(data.iloc[:,p]), mrmr_discretize(data.iloc[:,q])) for p in final_features_idx for q in final_features_idx])
        mrmr = 0
        for f in np.arange(chop_data.shape[1]):
            for ft in final_features_idx:
                D = (1/len(final_features_idx))*(D_sum + MIS(mrmr_discretize(chop_data.iloc[:,f]), labels))
                S = (1/(len(final_features_idx)**2))*(S_sum + MIS(mrmr_discretize(data.iloc[:,ft]),mrmr_discretize(chop_data.iloc[:,f])))
            score = D / S
            if score > mrmr:
                temp_idx = f
                mrmr = score
        final_features_idx.append(np.where(data.columns == chop_data.columns[temp_idx])[0][0])
        chop_data = chop_data.drop(chop_data.columns[temp_idx], axis = 1)
    return final_features_idx