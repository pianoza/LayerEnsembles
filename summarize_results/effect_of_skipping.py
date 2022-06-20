from re import A
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from pyparsing import alphanums
import seaborn as sns
import pickle
from sklearn.calibration import calibration_curve
import itertools
sns.set_theme(style='whitegrid')

sys.path.append('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method')
import configs
from utils import make_folders
from exp_list import get_experiments_list

def get_mean_std(orig_df):
    df = orig_df.copy()
    mask = df['MHD'] != np.inf
    df.loc[~mask, 'MHD'] = df.loc[mask, 'MHD'].max()
    general = {'DSC':['mean','std'],
                        'MHD':['mean','std'],
                        'NLL':['mean','std'],
                        # 'Entropy':['mean','std'],
                        # 'MI':['mean','std'],
                        # 'Variance':['mean','std'],
                        # 'AULA':['mean','std'],
                        }
    # multi_class = {'DSC-LV':['mean','std'], 'DSC-MYO':['mean','std'], 'DSC-RV':['mean','std'],
    #                'MHD-LV':['mean','std'], 'MHD-MYO':['mean','std'], 'MHD-RV':['mean','std'],
    #                'AULA-LV': ['mean','std'], 'AULA-MYO': ['mean','std'], 'AULA-RV': ['mean','std']}
    # if 'DSC-LV' in df.columns.tolist():
    #     general.update(multi_class)
    result = df.groupby(['Method'], sort=False, as_index=False).agg(general)
    return result

base_path = '/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/BCDR-TestRun-3-T1/segmentations/'
save_path = '/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/summary/'
filename = 'effect_of_skipping_layers.png'
df_all = []
for T in range(10, 0, -1):
    name_template = f'results_BCDR-TestRun-3-T{T}.csv'
    df_t = pd.read_csv(base_path + name_template)
    df_t['Method'] = ['Skip: '+str(10-T)] * len(df_t)
    df_t.rename(columns={'dsc_norm': 'DSC', 'mhd': 'MHD', 'nll': 'NLL'}, inplace=True)
    results = get_mean_std(df_t)
    df_all.append(results)
df1 = pd.concat(df_all, ignore_index=True)
print(df1)

base_path = '/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/MnM-TestRun-4-T1/segmentations/'
df_all = []
for T in range(10, 0, -1):
    name_template = f'results_MnM-TestRun-4-T{T}.csv'
    df_t = pd.read_csv(base_path + name_template)
    df_t['Method'] = ['Skip: '+str(10-T)] * len(df_t)
    df_t.rename(columns={'dsc_norm': 'DSC', 'mhd': 'MHD', 'nll': 'NLL', 'avg_variance': 'Variance', 'avg_entropy': 'Entropy', 'avg_mi': 'MI'}, inplace=True)
    results = get_mean_std(df_t)
    df_all.append(results)
df2 = pd.concat(df_all, ignore_index=True)
print(df2)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
LINE_WIDTH = 3

linecolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
ax[0].plot([0, 9], [0.615, 0.615], label='DE-BCDR', color='#1f77b4', linestyle='--', linewidth=LINE_WIDTH, alpha=0.8)
ax[1].plot([0, 9], [0.157, 0.157], label='DE-MnM', color='#1f77b4', linestyle='--', linewidth=LINE_WIDTH, alpha=0.8)
# BCDR
ax[0].plot(np.arange(0, 10), df1['NLL']['mean'], label='LE-BCDR', color='orange', linewidth=LINE_WIDTH)
lower_bound = [M_new - Sigma * 0.1 for M_new, Sigma in zip(df1['NLL']['mean'], df1['NLL']['std'])]
upper_bound = [M_new + Sigma * 0.1 for M_new, Sigma in zip(df1['NLL']['mean'], df1['NLL']['std'])]
ax[0].fill_between(np.arange(0, 10), lower_bound, upper_bound, alpha=.3)
# MnM
ax[1].plot(np.arange(0, 10), df2['NLL']['mean'], label='LE-MnM', color='orange', linewidth=LINE_WIDTH)
lower_bound = [M_new - Sigma * 0.1 for M_new, Sigma in zip(df2['NLL']['mean'], df2['NLL']['std'])]
upper_bound = [M_new + Sigma * 0.1 for M_new, Sigma in zip(df2['NLL']['mean'], df2['NLL']['std'])]
ax[1].fill_between(np.arange(0, 10), lower_bound, upper_bound, alpha=.3)
# Plot DSC values for each iteration
# BCDR
ax[0].text(9, 0.615, '{:.3f}\n({:.2f})'.format(0.870, 0.09), horizontalalignment='right', verticalalignment='bottom', fontsize=SMALL_SIZE)
for i in range(1, 10, 2):
    if i < 9:
        ax[0].text(i, df1['NLL']['mean'][i], '{:.3f}\n({:.2f})'.format(df1['DSC']['mean'][i], df1['DSC']['std'][i]), horizontalalignment='center', verticalalignment='bottom', fontsize=SMALL_SIZE)
    else:
        ax[0].text(i, df1['NLL']['mean'][i], '{:.3f}\n({:.2f})'.format(df1['DSC']['mean'][i], df1['DSC']['std'][i]), horizontalalignment='right', verticalalignment='top', fontsize=SMALL_SIZE)
# MnM
ax[1].text(9, 0.157, '{:.3f}\n({:.2f})'.format(0.896, 0.13), horizontalalignment='right', verticalalignment='bottom', fontsize=SMALL_SIZE)
for i in range(1, 10, 2):
    if i < 9:
        ax[1].text(i, df2['NLL']['mean'][i], '{:.3f}\n({:.2f})'.format(df2['DSC']['mean'][i], df2['DSC']['std'][i]), horizontalalignment='center', verticalalignment='bottom', fontsize=SMALL_SIZE)
    else:
        ax[1].text(i, df2['NLL']['mean'][i], '{:.3f}\n({:.2f})'.format(df2['DSC']['mean'][i], df2['DSC']['std'][i]), horizontalalignment='right', verticalalignment='top', fontsize=SMALL_SIZE)
for i, tick_step, tick_format in zip(range(2), [0.4, 0.05], ['%0.2f', '%0.2f']):
    ax[i].set_ylabel('NLL', fontsize=MEDIUM_SIZE)
    ax[i].set_xticks(np.arange(0, 10))
    ax[i].set_xlim([0, 9])
    ax[i].tick_params(labelsize=MEDIUM_SIZE)
    start, end = ax[i].get_ylim()
    ax[i].yaxis.set_ticks(np.arange(start, end, tick_step))
    ax[i].yaxis.set_major_formatter(ticker.FormatStrFormatter(tick_format))

ax[1].set_xlabel('Number of skipped layers', fontsize=MEDIUM_SIZE)
ax[0].legend(loc='upper left', fontsize=MEDIUM_SIZE)
ax[1].legend(loc='upper left', fontsize=MEDIUM_SIZE)
ax[0].set_title('Effect of skipping segmentation heads', size=BIGGER_SIZE)
plt.tight_layout(h_pad=0.4)
plt.savefig(save_path + filename, dpi=300)
plt.close()

