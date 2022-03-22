# RandomRad
### Premise
A literature review of 50 randomly-selected radiomic-machine learning publications indicated there are two broad areas of methodology manipulation which can inflate the performance and therefore significance of resultant models, tantamount to p-hacking. The experiments in this repository can be reproduced on other authentic (image-based) or random (interpolated) radiomic features to illustrate performance inflation from biased methodology. The two problem areas are: 
#### Inconsistent data partitioning
Problems with the split of available data into training, validation, test, and external test sets 
#### Unproductive feature associations
High-volume correlations between radiomic features and other high-dimensional variables including clinical factors, biological pathways, and the redundant association of radiomic features with themselves. 

### Cumulative result
Our study illustrates that radiomic p-hacking through inconsistent partitioning can result in a 1.4x performance boost, inflating performance from an average of 0.57 ROC-AUC to 0.80 ROC-AUC, coincidentally the same as the average ROC-AUC reported in the literature review.
![Supplement3](https://user-images.githubusercontent.com/51383554/159572496-1d462341-40c0-4976-9f74-e91457f2fa6d.jpg)


### Citation
M Gidwani, K Chang, J Patel, et al. Radiomic p-hacking: Inconsistent Partitioning and Unproductive Feature Associations. preprint (2022).
