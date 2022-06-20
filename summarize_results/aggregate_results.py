import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pickle
from sklearn import multiclass
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
                        'Entropy':['mean','std'],
                        'MI':['mean','std'],
                        'Variance':['mean','std'],
                        'AULA':['mean','std'],
                        }
    multi_class = {'DSC-LV':['mean','std'], 'DSC-MYO':['mean','std'], 'DSC-RV':['mean','std'],
                   'MHD-LV':['mean','std'], 'MHD-MYO':['mean','std'], 'MHD-RV':['mean','std'],
                   'AULA-LV': ['mean','std'], 'AULA-MYO': ['mean','std'], 'AULA-RV': ['mean','std']}
    if 'DSC-LV' in df.columns.tolist():
        general.update(multi_class)
    result = df.groupby(['Method'], sort=False, as_index=False).agg(general)
    return result

dataset = configs.DATASET
experiments = get_experiments_list(dataset)
if dataset == 'inbreast':
    plot_title = 'INbreast'
    boxplot_outname = 'INbreast-boxplot_all_metrics.png'
    all_metrics_csv = 'INbreast-table_all_metrics.csv'
    correlations_csv = 'INbreast-correlation.csv'
    drop_percentage_plot = 'INbreast-sorted_drop.png'
    calib_plot_out_name = 'Inbreast-calibration_plot.png'
    calib_plot_title = 'Calibration plot'
    dsc_threshold = 0.85  # considered a good segmentation
    seg_quality_control_plot = 'INbreast-seg_quality_control.png'
elif dataset == 'bcdr':
    plot_title = 'BCDR'
    boxplot_outname = 'BCDR-boxplot_all_metrics.png'
    all_metrics_csv = 'BCDR-table_all_metrics.csv'
    correlations_csv = 'BCDR-correlation.csv'
    drop_percentage_plot = 'BCDR-sorted_drop.png'
    calib_plot_out_name = 'BCDR-calibration_plot.png'
    calib_plot_title = 'Calibration plot'
    dsc_threshold = 0.95  # considered a good segmentation
    seg_quality_control_plot = 'BCDR-seg_quality_control.png'
    pd_plot_title = 'BCDR'
    prediction_depth_plot = 'BCDR-prediction_depth_plot.png'
elif dataset == 'mnm':
    plot_title = 'MnM'
    multi_class_plot = 'MnM-multi_class_plot.png'
    boxplot_outname = 'MnM-boxplot_all_metrics.png'
    all_metrics_csv = 'MnM-table_all_metrics.csv'
    correlations_csv = 'MnM-correlation.csv'
    multi_class_correlations_csv = 'MnM-multi_class_correlation.csv'
    drop_percentage_plot = 'MnM-sorted_drop.png'
    calib_plot_out_name = 'MnM-calibration_plot.png'
    calib_plot_title = 'Calibration plot'
    pd_plot_title = 'MnM'
    dsc_threshold = [0.90, 0.85, 0.80, 0.80]  # considered a good segmentation
    # dsc_threshold = [0.90, 0.85, 0.85, 0.85]  # considered a good segmentation
    seg_quality_control_plot = 'MnM-seg_quality_control_plot.png'
    prediction_depth_plot = 'MnM-prediction_depth_plot.png'
else:
    raise ValueError(f"{dataset} not implemented")

# load results per experiment
df_all = []
methods = []
multi_class = False
for method, exp_name in experiments.items():
    print('METHOD', method)
    print(exp_name)
    methods.append(method)
    models_path, figures_path, seg_out_path = make_folders(configs.RESULTS_PATH, exp_name)
    figures_save_path = figures_path
    df = pd.read_csv(f'{seg_out_path}/results_{exp_name}.csv')
    df.rename(columns={'dsc_norm': 'DSC', 'mhd': 'MHD', 'nll': 'NLL', 'avg_variance': 'Variance', 'avg_entropy': 'Entropy', 'avg_mi': 'MI', 'aula': 'AULA'}, inplace=True)
    # check if multi-class
    cols = df.columns.tolist()
    if 'dsc_cl1' in cols:
        multi_class = True
        df.rename(columns={'dsc_cl1': 'DSC-LV', 'dsc_cl2': 'DSC-MYO', 'dsc_cl3': 'DSC-RV', 'mhd_cl1': 'MHD-LV', 'mhd_cl2': 'MHD-MYO', 'mhd_cl3': 'MHD-RV',
                           'aula_cl1': 'AULA-LV', 'aula_cl2': 'AULA-MYO', 'aula_cl3': 'AULA-RV'}, inplace=True)
    df['Method'] = [method] * df.shape[0]
    df_all.append(df)

NUM_TOTAL_SAMPLES = df_all[0].shape[0]

df = pd.concat(df_all)
figures_save_path = configs.RESULTS_PATH / 'summary'
figures_save_path.mkdir(exist_ok=True, parents=False)

# BOXPLOT ALL METRICS
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
sns.boxplot(x='Method', y='DSC', data=df, showmeans=True, ax=ax[0, 0], showfliers=True,
            meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
sns.boxplot(x='Method', y='MHD', data=df, showmeans=True, ax=ax[0, 1], showfliers=True,
            meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
sns.boxplot(x='Method', y='NLL', data=df, showmeans=True, ax=ax[0, 2], showfliers=True,
            meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
# Do not plot uncertainty metrics for method Plain
dff = df[df['Method'] != 'Plain']
sns.boxplot(x='Method', y='Variance', data=dff, showmeans=True, ax=ax[1, 0], showfliers=True,
            meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
sns.boxplot(x='Method', y='Entropy', data=dff, showmeans=True, ax=ax[1, 1], showfliers=True,
            meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
sns.boxplot(x='Method', y='MI', data=dff, showmeans=True, ax=ax[1, 2], showfliers=True,
            meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
fig.suptitle(plot_title)

for i in range(2):
    for j in range(3):
        ax[i, j].set(xlabel=None)
plt.savefig(str(figures_save_path / boxplot_outname))

if multi_class:
    fig, ax = plt.subplots(3, 3, figsize=(20, 10))
    sns.boxplot(x='Method', y='DSC-LV', data=df, showmeans=True, ax=ax[0, 0], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.boxplot(x='Method', y='DSC-MYO', data=df, showmeans=True, ax=ax[0, 1], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.boxplot(x='Method', y='DSC-RV', data=df, showmeans=True, ax=ax[0, 2], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    # Do not plot uncertainty metrics for method Plain
    dff = df[df['Method'] != 'Plain']
    sns.boxplot(x='Method', y='MHD-LV', data=dff, showmeans=True, ax=ax[1, 0], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.boxplot(x='Method', y='MHD-MYO', data=dff, showmeans=True, ax=ax[1, 1], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.boxplot(x='Method', y='MHD-RV', data=dff, showmeans=True, ax=ax[1, 2], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.boxplot(x='Method', y='AULA-LV', data=dff, showmeans=True, ax=ax[2, 0], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.boxplot(x='Method', y='AULA-MYO', data=dff, showmeans=True, ax=ax[2, 1], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    sns.boxplot(x='Method', y='AULA-RV', data=dff, showmeans=True, ax=ax[2, 2], showfliers=True,
                meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
    fig.suptitle(plot_title)

    for i in range(2):
        for j in range(3):
            ax[i, j].set(xlabel=None)
    plt.savefig(str(figures_save_path / multi_class_plot))


# MEAN AND STD FOR ALL METRICS
result = get_mean_std(df)
print(result)
result.to_csv(str(figures_save_path / all_metrics_csv), index=False)

# CORRELATION FOR TWO METRICS
# mask = df['MHD'] != np.inf
# df.loc[~mask, 'MHD'] = df.loc[mask, 'MHD'].max()
result = df.groupby('Method', sort=False)[['DSC', 'MHD', 'Entropy', 'MI', 'Variance', 'AULA']].corr(method='spearman')
# result = df.groupby('Method', sort=False)[['DSC', 'MHD', 'Entropy', 'MI', 'Variance', 'AULA']].corr(method='pearson')
result = result[['Entropy', 'MI', 'Variance', 'AULA']]
result = result.drop(['Entropy', 'MI', 'Variance', 'AULA'], level=1)
print(result)
result.to_csv(str(figures_save_path / correlations_csv))

if multi_class:
    multi = ['DSC-LV', 'DSC-MYO', 'DSC-RV', 'MHD-LV', 'MHD-MYO', 'MHD-RV', 'AULA-LV', 'AULA-MYO', 'AULA-RV']
    result = df.groupby('Method', sort=False)[multi].corr(method='spearman')
    result = result[multi[6:]]
    result = result.drop(multi[6:], level=1)
    print(result)
    result.to_csv(str(figures_save_path / multi_class_correlations_csv))

# PERCENTAGE REMOVAL OF MOST UNCERTAIN SAMPLES
# sorted_by_entropy = df.sort_values(['Entropy'], ascending=True).groupby('Method')
# sorted_by_entropy = df.sort_values(['MI'], ascending=True).groupby('Method')
# sorted_by_entropy = df.sort_values(['Variance'], ascending=True).groupby('Method')
sorted_by_entropy = df.sort_values(['AULA'], ascending=True).groupby('Method')
drop_percentage = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
mean = []
std = []
met = [] # method 
dp = []  # drop percentage
for part in drop_percentage:
    count_head = NUM_TOTAL_SAMPLES - int((NUM_TOTAL_SAMPLES * part) / 100)
    print(f'Number of remaining samples {count_head}')
    head = sorted_by_entropy.head(count_head)
    # head = df.groupby('Method').sample(n=count_head)
    mean_std = get_mean_std(head)
    for method in methods:
        method_ms = mean_std.loc[mean_std['Method'] == method]['DSC']
        mean.append(method_ms['mean'].item())
        std.append(method_ms['std'].item())
        met.append(method)
        dp.append(part)
perc_df = {
    'Mean': mean,
    'Std': std,
    'Method': met,
    'Drop': dp
}
perc_df = pd.DataFrame(perc_df)
print(perc_df)
fig, ax = plt.subplots(1, 1)
for method in methods:
    lower_bound = [M_new - Sigma * 0.1 for M_new, Sigma, Method in zip(mean, std, met) if Method == method]
    upper_bound = [M_new + Sigma * 0.1 for M_new, Sigma, Method in zip(mean, std, met) if Method == method]
    ax.fill_between(drop_percentage, lower_bound, upper_bound, alpha=.1)
sns.lineplot(data=perc_df, x='Drop', y='Mean', hue='Method', linewidth=2., ax=ax)
ax.set_title('DSC')
ax.set_xlabel('% of dropped samples')
plt.savefig(str(figures_save_path / drop_percentage_plot))

# SEGMENTATION QUALITY CONTROL PLOTS
# x - number of images flagged for manual correction
# y - number of remaining poor segmentation quality images
# sorted_by_unc = df.sort_values(['Entropy'], ascending=True).groupby('Method')
# sorted_by_unc = df.sort_values(['MI'], ascending=True).groupby('Method')
# sorted_by_unc = df.sort_values(['Variance'], ascending=True).groupby('Method')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
LINE_WIDTH = 3

if not multi_class:
    sorted_by_unc = df.sort_values(['AULA'], ascending=False).groupby('Method').head(NUM_TOTAL_SAMPLES)
    sorted_by_dsc = df.sort_values(['DSC'], ascending=True).groupby('Method').head(NUM_TOTAL_SAMPLES)
    fraction_flagged_manual_correction = np.linspace(0, 1, 50)
    auc_dx = fraction_flagged_manual_correction[1] - fraction_flagged_manual_correction[0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    max_remaining = 0
    ideal_max_remaining = 0
    ideal_min_remaining = np.inf
    ideal_min_remaining_dropped_fraction = 0

    max_method_name_lenght = len(max(methods[1:], key=len))  # skip Plain 
    for method in methods:
        if 'Plain' in method: continue
        fraction_poor_segmentation_remaining = []
        ideal_fraction_poor_segmentation_remaining = []
        # ideal_fraction_of_flagged_images = []
        for drop_fraction in fraction_flagged_manual_correction:
            drop_quantity = int(NUM_TOTAL_SAMPLES * drop_fraction)

            method_samples = sorted_by_unc[sorted_by_unc['Method'] == method]
            method_samples_ideal = sorted_by_dsc[sorted_by_dsc['Method'] == method]

            remaining_rows = method_samples.iloc[drop_quantity:]
            num_remaining_poor_segmentation = remaining_rows[remaining_rows['DSC'] < dsc_threshold].shape[0]
            fraction_poor_segmentation_remaining.append(num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)
            max_remaining = max(max_remaining, num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)

            # how many images should we flag for manual correction to have the same as num_remaining_poor_segmentation samples?
            remaining_rows_ideal = method_samples_ideal.iloc[drop_quantity:]
            ideal_num_remaining_poor_segmentation = remaining_rows_ideal[remaining_rows_ideal['DSC'] < dsc_threshold].shape[0]
            ideal_fraction_poor_segmentation_remaining.append(ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)

            if drop_fraction == 0:
                ideal_max_remaining += ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES
            if ideal_min_remaining > ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES:
                ideal_min_remaining = ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES
                ideal_min_remaining_dropped_fraction = drop_fraction

        auc = np.trapz(fraction_poor_segmentation_remaining, dx=auc_dx)
        method_name_lenght = len(method)
        gap = ' ' * (max_method_name_lenght - method_name_lenght)
        ax.plot(fraction_flagged_manual_correction, fraction_poor_segmentation_remaining, label=method+gap+", AUC="+str(round(auc, 3)), linewidth=LINE_WIDTH)

    # Ideal line when the samples are dropped by DSC as opposed to dropping samples by uncertainty
    # auc = np.trapz(np.linspace(ideal_max_remaining / len(methods), ideal_min_remaining, 50), dx=auc_dx)
    ax.fill_between([0, ideal_min_remaining_dropped_fraction], [ideal_max_remaining / len(methods[1:]), ideal_min_remaining], color='lightgrey') #, label='Ideal'+", AUC="+str(round(auc, 3)))
    # 5% poor segmentation remaining horizontal line
    ax.plot(fraction_flagged_manual_correction, np.ones(len(fraction_flagged_manual_correction)) * 0.05, ':', color='darkgrey', linewidth=LINE_WIDTH)
    # Random drop
    # auc = np.trapz(np.linspace(max_remaining, 0, 50), dx=auc_dx)
    ax.plot([0, 1], [max_remaining, 0], 'k--', linewidth=LINE_WIDTH)  #, label='Random'+", AUC="+str(round(auc, 3)))
    # ax.set_title('Segmentation quality control')
    ax.set_title('BCDR', fontsize=BIGGER_SIZE)
    ax.set_xlabel('Flagged for manual correction', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Remaining poor segmentation fraction', fontsize=MEDIUM_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(ymin=0)
    ax.tick_params(labelsize=MEDIUM_SIZE)
    ax.legend(prop={'family': 'monospace', 'size': MEDIUM_SIZE})
    # # get the width of your widest label, since every label will need 
    # # to shift by this amount after we align to the right
    # renderer = fig.canvas.get_renderer()
    # legend = ax.get_legend()
    # shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
    # for t in legend.get_texts():
    #     t.set_ha('right') # ha is alias for horizontalalignment
    #     t.set_position((shift,0))
    plt.tight_layout(h_pad=0.4)
    plt.savefig(str(figures_save_path / seg_quality_control_plot), dpi=300)
else:
    # COMBINED AND ALL CLASSES TOGETHER
    line_styles = ['-', '--', '-.', ':']
    colors = {
        'DE': '#1f77b4',
        'LE': 'orange',
    }
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fraction_flagged_manual_correction = np.linspace(0, 1, 50)
    auc_dx = fraction_flagged_manual_correction[1] - fraction_flagged_manual_correction[0]
    max_remaining = 0
    ideal_max_remaining = 0
    ideal_min_remaining = np.inf
    ideal_min_remaining_dropped_fraction = 0

    for idx, cl in enumerate(['', '-LV', '-MYO', '-RV']):
        sorted_by_unc = df.sort_values(['AULA'+cl], ascending=False).groupby('Method').head(NUM_TOTAL_SAMPLES)
        sorted_by_dsc = df.sort_values(['DSC'+cl], ascending=True).groupby('Method').head(NUM_TOTAL_SAMPLES)

        max_method_name_lenght = len(max(methods[1:], key=len))+4  # skip Plain + 4 for -MYO
        for method in methods:
            if 'Plain' in method: continue
            fraction_poor_segmentation_remaining = []
            ideal_fraction_poor_segmentation_remaining = []
            # ideal_fraction_of_flagged_images = []
            for drop_fraction in fraction_flagged_manual_correction:
                drop_quantity = int(NUM_TOTAL_SAMPLES * drop_fraction)

                method_samples = sorted_by_unc[sorted_by_unc['Method'] == method]
                method_samples_ideal = sorted_by_dsc[sorted_by_dsc['Method'] == method]

                remaining_rows = method_samples.iloc[drop_quantity:]
                num_remaining_poor_segmentation = remaining_rows[remaining_rows['DSC'+cl] < dsc_threshold[idx]].shape[0]
                fraction_poor_segmentation_remaining.append(num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)
                if idx == 0:
                    max_remaining = max(max_remaining, num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)

                # how many images should we flag for manual correction to have the same as num_remaining_poor_segmentation samples?
                remaining_rows_ideal = method_samples_ideal.iloc[drop_quantity:]
                ideal_num_remaining_poor_segmentation = remaining_rows_ideal[remaining_rows_ideal['DSC'+cl] < dsc_threshold[idx]].shape[0]
                ideal_fraction_poor_segmentation_remaining.append(ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)
            
                if idx == 0:
                    if drop_fraction == 0:
                        ideal_max_remaining = max(ideal_max_remaining, ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)
                    if ideal_min_remaining > ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES:
                        ideal_min_remaining = ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES
                        ideal_min_remaining_dropped_fraction = drop_fraction

            auc = np.trapz(fraction_poor_segmentation_remaining, dx=auc_dx)
            method_name_lenght = len(method+cl)
            gap = ' ' * (max_method_name_lenght - method_name_lenght)
            ax.plot(fraction_flagged_manual_correction, fraction_poor_segmentation_remaining, label=method+cl+gap+", AUC="+str(round(auc, 3)), linewidth=LINE_WIDTH, color=colors[method], linestyle=line_styles[idx])

    # Ideal line when the samples are dropped by DSC as opposed to dropping samples by uncertainty
    # auc = np.trapz(np.linspace(ideal_max_remaining / len(methods), ideal_min_remaining, 50), dx=auc_dx)
    ax.fill_between([0, ideal_min_remaining_dropped_fraction], [ideal_max_remaining, ideal_min_remaining], color='lightgrey') #, label='Ideal'+", AUC="+str(round(auc, 3)))
    # 5% poor segmentation remaining horizontal line
    ax.plot(fraction_flagged_manual_correction, np.ones(len(fraction_flagged_manual_correction)) * 0.05, ':', color='darkgrey', linewidth=LINE_WIDTH)
    # Random drop
    # auc = np.trapz(np.linspace(max_remaining, 0, 50), dx=auc_dx)
    ax.plot([0, 1], [max_remaining, 0], 'k--', linewidth=LINE_WIDTH)  #, label='Random'+", AUC="+str(round(auc, 3)))
    # ax[idx].set_title('Segmentation quality control {}'.format(cl[1:]))
    ax.set_title('MnM', fontsize=BIGGER_SIZE)
    ax.set_xlabel('Flagged for manual correction', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Remaining poor segmentation fraction', fontsize=MEDIUM_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(ymin=0)
    ax.tick_params(labelsize=MEDIUM_SIZE)
    ax.legend(prop={'family': 'monospace', 'size': MEDIUM_SIZE})
    # # get the width of your widest label, since every label will need 
    # # to shift by this amount after we align to the right
    # renderer = fig.canvas.get_renderer()
    # legend = ax.get_legend()
    # shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
    # for t in legend.get_texts():
    #     t.set_ha('right') # ha is alias for horizontalalignment
    #     t.set_position((shift,0))
    # fig.text(0.5, 0.05, 'Fraction of images flagged for manual correction', ha='center')
    plt.tight_layout(h_pad=4)
    plt.savefig(str(figures_save_path / seg_quality_control_plot), dpi=300)
    plt.close()
    # EACH CLASS SEPARATELY
    # fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    # for idx, cl in enumerate(['', '-LV', '-MYO', '-RV']):
    #     sorted_by_unc = df.sort_values(['AULA'+cl], ascending=False).groupby('Method').head(NUM_TOTAL_SAMPLES)
    #     sorted_by_dsc = df.sort_values(['DSC'+cl], ascending=True).groupby('Method').head(NUM_TOTAL_SAMPLES)
    #     fraction_flagged_manual_correction = np.linspace(0, 1, 50)
    #     auc_dx = fraction_flagged_manual_correction[1] - fraction_flagged_manual_correction[0]
    #     max_remaining = 0
    #     ideal_max_remaining = 0
    #     ideal_min_remaining = np.inf
    #     ideal_min_remaining_dropped_fraction = 0

    #     max_method_name_lenght = len(max(methods[1:], key=len))  # skip Plain 
    #     for method in methods:
    #         if 'Plain' in method: continue
    #         fraction_poor_segmentation_remaining = []
    #         ideal_fraction_poor_segmentation_remaining = []
    #         # ideal_fraction_of_flagged_images = []
    #         for drop_fraction in fraction_flagged_manual_correction:
    #             drop_quantity = int(NUM_TOTAL_SAMPLES * drop_fraction)

    #             method_samples = sorted_by_unc[sorted_by_unc['Method'] == method]
    #             method_samples_ideal = sorted_by_dsc[sorted_by_dsc['Method'] == method]

    #             remaining_rows = method_samples.iloc[drop_quantity:]
    #             num_remaining_poor_segmentation = remaining_rows[remaining_rows['DSC'+cl] < dsc_threshold[idx]].shape[0]
    #             fraction_poor_segmentation_remaining.append(num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)
    #             max_remaining = max(max_remaining, num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)

    #             # how many images should we flag for manual correction to have the same as num_remaining_poor_segmentation samples?
    #             remaining_rows_ideal = method_samples_ideal.iloc[drop_quantity:]
    #             ideal_num_remaining_poor_segmentation = remaining_rows_ideal[remaining_rows_ideal['DSC'+cl] < dsc_threshold[idx]].shape[0]
    #             ideal_fraction_poor_segmentation_remaining.append(ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES)

    #             if drop_fraction == 0:
    #                 ideal_max_remaining += ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES
    #             if ideal_min_remaining > ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES:
    #                 ideal_min_remaining = ideal_num_remaining_poor_segmentation / NUM_TOTAL_SAMPLES
    #                 ideal_min_remaining_dropped_fraction = drop_fraction

    #         auc = np.trapz(fraction_poor_segmentation_remaining, dx=auc_dx)
    #         method_name_lenght = len(method)
    #         gap = ' ' * (max_method_name_lenght - method_name_lenght)
    #         ax[idx].plot(fraction_flagged_manual_correction, fraction_poor_segmentation_remaining, label=method+gap+", AUC="+str(round(auc, 3)), linewidth=2.)

    #     # Ideal line when the samples are dropped by DSC as opposed to dropping samples by uncertainty
    #     # auc = np.trapz(np.linspace(ideal_max_remaining / len(methods), ideal_min_remaining, 50), dx=auc_dx)
    #     ax[idx].fill_between([0, ideal_min_remaining_dropped_fraction], [ideal_max_remaining / len(methods[1:]), ideal_min_remaining], color='lightgrey') #, label='Ideal'+", AUC="+str(round(auc, 3)))
    #     # 5% poor segmentation remaining horizontal line
    #     ax[idx].plot(fraction_flagged_manual_correction, np.ones(len(fraction_flagged_manual_correction)) * 0.05, ':', color='darkgrey')
    #     # Random drop
    #     # auc = np.trapz(np.linspace(max_remaining, 0, 50), dx=auc_dx)
    #     ax[idx].plot([0, 1], [max_remaining, 0], 'k--')  #, label='Random'+", AUC="+str(round(auc, 3)))
    #     # ax[idx].set_title('Segmentation quality control {}'.format(cl[1:]))
    #     ax[idx].set_title('MnM {}'.format(cl[1:]))
    #     ax[idx].set_xlabel(' ')
    #     if idx == 0:
    #         ax[idx].set_ylabel('Fraction of images with poor segmentation')
    #     ax[idx].set_xlim(0, 1)
    #     ax[idx].set_ylim(ymin=0)
    #     ax[idx].legend(prop={'family': 'monospace'})
    #     # # get the width of your widest label, since every label will need 
    #     # # to shift by this amount after we align to the right
    #     # renderer = fig.canvas.get_renderer()
    #     # legend = ax.get_legend()
    #     # shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
    #     # for t in legend.get_texts():
    #     #     t.set_ha('right') # ha is alias for horizontalalignment
    #     #     t.set_position((shift,0))
    # fig.text(0.5, 0.05, 'Fraction of images flagged for manual correction', ha='center')
    # plt.tight_layout(h_pad=4)
    # plt.savefig(str(figures_save_path / seg_quality_control_plot), dpi=300)

# def get_prediction_depth(agreement, threshold):
#     pd = len(agreement)  # initially, we set the prediction depth to the highest
#     for i in range(len(agreement)-1, 0, -1):
#         delta = np.abs(agreement[i] - agreement[i-1])
#         if delta <= 1 - threshold:  # threshold is dsc, e.g., 0.9 for 90% agreement
#             pd = i-1 # then we set the prediction depth to the lowest
#         else:
#             break
#     return pd

# # Prediction depth plot
# # if False and multi_class:
# methods = list()
# fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
# colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'black']
# prediction_depths_all = {
#     "Prediction Depth": [],
#     "Frequency": [],
#     "Method": []
# }
# for i, (method, exp_name) in enumerate(experiments.items()):
#     methods.append(method)
#     models_path, figures_path, seg_out_path = make_folders(configs.RESULTS_PATH, exp_name)
#     print(method)
#     pd_path = f'results_{exp_name}-prediction_depth_all.pkl'
#     pickled_prediction_depths = pickle.load( open(str(seg_out_path / pd_path), "rb"))  # (N, T)
#     # calculate prediction depth for each sample in prediction_depth_all NOTE this threshold is hardcoded
#     prediction_depths = [get_prediction_depth(agreement, 0.95) for agreement in pickled_prediction_depths]
#     # calculate the histogram
#     hist, bins = np.histogram(prediction_depths, bins=np.arange(1, 11, 1), density=True)
#     prediction_depths_all["Prediction Depth"].extend(bins[1:])
#     prediction_depths_all["Frequency"].extend(hist)
#     prediction_depths_all["Method"].extend([method] * len(bins[1:]))
#     # plot a bar chart with the histogram
#     # ax.bar(bins[:-1]+bar_width*i, hist, width=bar_width, color=colors[i], label=method)
# sns.barplot(x="Prediction Depth", y="Frequency", hue="Method", data=prediction_depths_all, ax=ax)
# plt.title(pd_plot_title)
# plt.xlabel('Prediction depth')
# plt.ylabel('Frequency')
# plt.xticks(np.arange(0, 9, 1), ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10'])
# plt.legend(prop={'size': 10})
# plt.tight_layout()
# figures_save_path = configs.RESULTS_PATH / 'summary'
# figures_save_path.mkdir(parents=True, exist_ok=True)
# plt.savefig(str(figures_save_path / prediction_depth_plot), dpi=300)


# # Calibration plot
# # load results per experiment
# if False:
#     if not multi_class:  # calibration plot doesn't make sense for multi-class
#         methods = list()
#         y_true = dict()
#         y_prob = dict()
#         fig, ax = plt.subplots(1, 1)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.plot([0, 1], [0, 1], 'k--')
#         for method, exp_name in experiments.items():
#             methods.append(method)
#             models_path, figures_path, seg_out_path = make_folders(configs.RESULTS_PATH, exp_name)
#             print(method)
#             y_true[method] = []
#             y_prob[method] = []
#             probas_path = f'results_{exp_name}-calibration_pairs.pkl'
#             # calibration pairs is a list of tuples of (lbl, probs)
#             calibration_pairs = pickle.load( open(str(seg_out_path / probas_path), "rb"))
#             y_true[method].extend([a[0] for a in calibration_pairs])
#             y_prob[method].extend([a[1] for a in calibration_pairs])
#             yt = list(itertools.chain.from_iterable([x.flatten() for x in y_true[method]]))
#             yp = list(itertools.chain.from_iterable([x.flatten() for x in y_prob[method]]))
#             prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10)
#             ax.plot(prob_pred, prob_true, label=method)
#             # ax.bar(prob_pred, prob_true, width=0.01, label=method)
#         plt.title(calib_plot_title)
#         plt.xlabel('Predicted probabilities')
#         plt.ylabel('Observed probabilities')
#         plt.legend()
#         figures_save_path = configs.RESULTS_PATH / 'summary'
#         figures_save_path.mkdir(parents=True, exist_ok=True)
#         plt.tight_layout()
#         plt.savefig(str(figures_save_path / calib_plot_out_name))