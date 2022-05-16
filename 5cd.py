import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from MRMR import mrmr
import gseapy
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


testset_smote = np.load('randomrad_smote.npy')

def dataset_norm(x):
	return (x - np.mean(x))/np.std(x)
def train_split_norm(x,y):
	return (x - np.mean(y))/np.std(y)
	
testset_smote_norm = testset_smote.copy()
#we will now normalize all the data together to demonstrate the unfair advatage that confers on further experimention 
for column in range(testset_smote_norm.shape[1]):
	testset_smote_norm[:,column] = dataset_norm(testset_smote[:,column])

clinical_vars = pd.read_csv('ClinicalData/TCGA_LGG_clinical_data.csv', index_col = 1) #TCGA Clinical Data downloaded from CBioPortal
clinical_vars = clinical_vars.loc[~clinical_vars.index.duplicated(keep='first')]
LGG_annotated_cases = np.load('ClinicalData/LGG_annotated_cases.npy', allow_pickle = True)

OS = np.array(clinical_vars.loc[LGG_annotated_cases,'Overall Survival (Months)'])
OS_severity = OS>=np.median(OS)
OS_severity = OS_severity*1
OS_censor_flag = (clinical_vars.loc[LGG_annotated_cases,'Overall Survival Status'] == '1:DECEASED').astype(int).values

"""
FPKM_folders = sorted(os.listdir('transcriptome_profiling'))
basedir = os.getcwd()
os.chdir(os.path.join(basedir,'transcriptome_profiling',os.listdir('transcriptome_profiling')[0]))
TP_filename = [g for g in os.listdir(os.getcwd()) if '.FPKM.txt' in g][0]
TCGA_LGG_expression_df = pd.read_csv(TP_filename, index_col = 0, sep = '\t', header = None)
os.chdir(basedir)
count = 1
for folder in FPKM_folders[1:]:
	count = count + 1
	TP_filename = [g for g in os.listdir(os.path.join(basedir,'transcriptome_profiling',folder)) if '.FPKM.txt' in g][0]
	temp_expression_df = pd.read_csv(os.path.join(basedir,'transcriptome_profiling',folder,TP_filename), index_col = 0, sep = '\t', header = None)
	TCGA_LGG_expression_df[count] = ""
	TCGA_LGG_expression_df.loc[temp_expression_df.index,count] = temp_expression_df[1] 
TCGA_LGG_expression_df.columns = FPKM_folders
par_locus_y_chromosome_genes = [g for g in TCGA_LGG_expression_df.index.values if 'ENSGR' in g]
TCGA_LGG_expression_df = TCGA_LGG_expression_df.drop(par_locus_y_chromosome_genes)
feature_cases = np.array(LGG_annotated_cases)
feature_cases = np.delete(feature_cases,[np.where(feature_cases == p)[0][0] for p in ['TCGA-CS-6665','TCGA-DU-5851']]) #remove cases which don't have FPKM data
TCGA_LGG_expression_df = TCGA_LGG_expression_df.T.loc[feature_cases].T #only choosing cases with features and in the correct order
TCGA_LGG_expression_df = TCGA_LGG_expression_df.reset_index()
TCGA_LGG_expression_df.rename(columns={0:'NAME'}, inplace=True)
genekey = pd.read_csv('ENSEMBL80_genes.txt', index_col = 0)
gene_names = []
for name in TCGA_LGG_expression_df.NAME.values:
	try:
		genekey.loc[np.char.split(name,'.').tolist()[0] ,'Associated Gene Name'].shape
		gene_names.append(genekey.loc[np.char.split(name,'.').tolist()[0] ,'Associated Gene Name'].drop_duplicates().values[0])
	except:
		gene_names.append(genekey.loc[np.char.split(name,'.').tolist()[0] ,'Associated Gene Name'])
TCGA_LGG_expression_df['Gene Name'] = ""
TCGA_LGG_expression_df['Gene Name'] = gene_names
#There are some repeated genes in our dataframe because ENSEMBL IDs are unique identifiers, while gene names are not. 
#Since these are transcript expression levels, we can sum over the repeated gene names.
TCGA_LGG_expression_df = TCGA_LGG_expression_df.groupby('Gene Name')[TCGA_LGG_expression_df.columns[1:-1]].sum()
TCGA_LGG_expression_df = TCGA_LGG_expression_df.reset_index()

gsea_ready_OS_severity = list(np.delete(OS_severity,[np.where(LGG_annotated_cases == p)[0][0] for p in ['TCGA-CS-6665','TCGA-DU-5851']])) #remove the cases without FPKM data
gsea_LGG = gseapy.gsea(data=TCGA_LGG_expression_df, 
					   gene_sets='KEGG_2016', 
					   cls = gsea_ready_OS_severity,
					   permutation_type='phenotype', 
					   outdir=None, 
					   method='signal_to_noise')
ssgsea_LGG_radiomics = gseapy.ssgsea(data=TCGA_LGG_expression_df,
								 gene_sets='KEGG_2016',
								 outdir=None,
								 sample_norm_method='log_rank', 
								 permutation_num=0,
								 no_plot=True) 
"""
ssgsea_LGG_radiomics = pd.read_csv('ssgsea_LGG_radiomics.csv',index_col=0)
gsea_LGG = pd.read_csv('gsea_LGG.csv',index_col=0)
rad_gene_set_pvalues = np.zeros((ssgsea_LGG_radiomics.shape[0], testset_smote.shape[1])) #res2d
for i in np.arange(testset_smote.shape[1]):
	for j in np.arange(ssgsea_LGG_radiomics.shape[0]): #res2d
		rad_gene_set_pvalues[j,i] = spearmanr(ssgsea_LGG_radiomics.iloc[j], #res2d
											  np.delete(testset_smote, [np.where(LGG_annotated_cases == p)[0][0] for p in ['TCGA-CS-6665','TCGA-DU-5851']], axis = 0)[:,i])[1]
print('Number of significant feature-gene associations: ',sum(sum(rad_gene_set_pvalues < (0.1/(rad_gene_set_pvalues.shape[0]*rad_gene_set_pvalues.shape[1]))))) #Bonferroni correction
print('Without correcting for multiple hypothesis testing, ',np.unique(np.where(rad_gene_set_pvalues < 0.1)[0]).shape[0],' of ',rad_gene_set_pvalues.shape[0],' gene sets have an association with at least one fake radiomic feature.')
GO_names = ssgsea_LGG_radiomics.iloc[np.unique(np.where(rad_gene_set_pvalues < (0.1/(rad_gene_set_pvalues.shape[0]*rad_gene_set_pvalues.shape[1])))[0])].index #res2d
counts = np.unique(np.where(rad_gene_set_pvalues < (0.1/(rad_gene_set_pvalues.shape[0]*rad_gene_set_pvalues.shape[1])))[0], return_counts = True)[1]
for i in range(len(GO_names)):
    plt.barh(GO_names[i],counts[i])
plt.xlabel('Number of Radiomic Phenotype Associations')
plt.title('Molecular Basis for Radiomic Features')
plt.show()

for ft in np.unique(np.where(rad_gene_set_pvalues < (0.1/(rad_gene_set_pvalues.shape[0]*rad_gene_set_pvalues.shape[1])))[1]): #fake rad fts with sig assns
    #split the samples into 2 groups based on the median value of the feature
    i1 = np.where(testset_smote_norm[:,ft]  >= np.median(testset_smote_norm[:,ft]))[0]
    i2 = np.where(testset_smote_norm[:,ft]  < np.median(testset_smote_norm[:,ft]))[0]
    result = logrank_test(OS[i1], OS[i2], OS_censor_flag[i1], OS_censor_flag[i2])
    if result.p_value < 0.05:
        break
i1 = np.where(testset_smote_norm[:,ft]  >= np.median(testset_smote_norm[:,ft]))[0]
i2 = np.where(testset_smote_norm[:,ft]  < np.median(testset_smote_norm[:,ft]))[0]
result = logrank_test(OS[i1], OS[i2], OS_censor_flag[i1], OS_censor_flag[i2])

kmf = KaplanMeierFitter()
kmf.fit(OS[i1], OS_censor_flag[i1], label = 'high feature value')
a1 = kmf.plot()

kmf.fit(OS[i2], OS_censor_flag[i2], label = 'low feature value')
kmf.plot(ax=a1)

plt.title('Overall Survival split by random feature associated with GO pathway')
plt.text(0,0,'Logrank test statistic = '+ str(round(result.test_statistic,3)))
plt.text(0,0.1,'Logrank p-value = ' + str(round(result.p_value,3)))
plt.show()