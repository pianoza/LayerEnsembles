import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pylab import *
from scipy.optimize import curve_fit

# # Training uncertainty
# uncertainty_file = Path('/home/kaisar/EuCanImage/Coding/UQComparison/mass_segmentation/results/summary/TRAINING_UNCERTAINTIES.csv')
# OPTIMAM
uncertainty_file = Path('/home/kaisar/Datasets/OPTIMAM/optimam_info_fixed.csv')
# Test set
# uncertainty_file = Path('/home/kaisar/EuCanImage/Coding/UQComparison/mass_segmentation/results/SVGD-BCDR-SelfTraining/segmentations/results_SVGD-BCDR-SelfTraining.csv')

# # joint distribution of training uncertainties in training samples after using SVGD-SelfTraining
# uncertainties = pd.read_csv(uncertainty_file)
# # uncertainties = pd
# fig, ax = plt.subplots(1, 3)

# g1 = sns.JointGrid(x='avg_mi', y='avg_entropy', data=uncertainties)
# g1.plot_joint(sns.scatterplot, alpha=0.5, edgecolor='.2', linewidth=.5)
# g1.plot_marginals(sns.histplot, kde=True)

# g2 = sns.JointGrid(x='avg_mi', y='avg_variance', data=uncertainties)
# g2.plot_joint(sns.scatterplot, alpha=0.5, edgecolor='.2', linewidth=.5)
# g2.plot_marginals(sns.histplot, kde=True)

# g3 = sns.JointGrid(x='avg_variance', y='avg_entropy', data=uncertainties)
# g3.plot_joint(sns.scatterplot, alpha=0.5, edgecolor='.2', linewidth=.5)
# g3.plot_marginals(sns.histplot, kde=True)

# plt.show()



uncertainties = pd.read_csv(uncertainty_file)

data = uncertainties['avg_mi'].values

y, x, _ = hist(data, 100, alpha=.3, label='unc metric')

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

# expected=(1,.2,250,2,.2,125)
# params,cov=curve_fit(bimodal,x,y,expected)
# sigma=sqrt(diag(cov))
# plot(x,bimodal(x,*params),color='red',lw=3,label='model')
# legend()
# print(params,'\n',sigma)    