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

sys.path.append('/home/kaisar/EuCanImage/Coding/LayerEnsembles')
import configs
from utils import make_folders
from exp_list import get_experiments_list

def run_make_summary():
    # prep
    experiments = get_experiments_list(configs.DATASET)
    df, methods, num_classes = load_experiments(experiments)
    save_path = configs.RESULTS_PATH / 'summary'
    save_path.mkdir(exist_ok=True, parents=False)
    # mean and std per metric for all methods
    result = get_mean_and_std(df, num_classes)
    result.to_csv(save_path / 'mean_and_std.csv', index=False)
    # correlation for all methods
    result = get_correlation(df, num_classes, 'spearman')
    result.to_csv(save_path / 'correlation.csv')
    # quality control
    plot_quality_control(df, num_classes, save_path / 'quality_control.png')

def plot_quality_control(df, num_classes, save_path):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    LINE_WIDTH = 3
    # COMBINED AND ALL CLASSES TOGETHER
    line_styles = ['-', '--', '-.', ':']
    colors = [
        #afe3c0, #a8dbb4, #a0d3a8, #90c290, #82ba96,
        #73b19c, #559fa7, #197bbd, #688b58, #3f3f37
    ]
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
    pass

def get_correlation(df, num_classes, corr_type):
    # remove Plain
    df = df[df['Method'] != 'Plain']
    # get correlation for all methods
    unc_targets = ['variance', 'entropy', 'mi']
    perf_targets = []
    for i in range(1, num_classes):
        unc_targets.append(f'aula_{i}')
        perf_targets.extend([f'dsc_{i}', f'hd_{i}', f'mhd_{i}'])
    targets= unc_targets + perf_targets
    result = df.groupby(['Method'], sort=False)[targets].corr(method=corr_type)
    result = result[unc_targets]
    result = result.drop(unc_targets, level=1)
    return result


def get_mean_and_std(df, num_classes):
    # create signature dictionary
    sample_level = {'entropy', 'variance', 'mi', 'nll'}
    signature = dict()
    col_names = set([col.split('_')[0] for col in df.columns.tolist()])
    col_names.remove('Method')
    for col_name in col_names:
        for i in range(1, num_classes):
            col_name = f'{col_name}_{i}' if col_name not in sample_level else col_name
            signature[col_name] = ['mean', 'std']
    result = df.groupby(['Method'], sort=False, as_index=False).agg(signature)
    first_cols = [('Method', ''),
                  ('entropy', 'mean'), ('entropy', 'std'),
                  ('variance', 'mean'), ('variance', 'std'),
                  ('mi', 'mean'), ('mi', 'std'),
                  ('nll', 'mean'), ('nll', 'std')]
    last_cols = [col for col in result.columns if col not in first_cols]
    result = result[first_cols + last_cols]
    return result

def load_experiments(experiments):
    # load results per experiment
    df_all = []
    methods = []
    for method, exp_name in experiments.items():
        print('METHOD', method)
        print(exp_name)
        methods.append(method)
        _, _, metrics_out_path = make_folders(configs.RESULTS_PATH, exp_name)
        df = pd.read_csv(f'{metrics_out_path}/results_{exp_name}.csv', index_col=0)
        col_names = df.columns.tolist()
        # get number of classes
        num_classes = len([col for col in col_names if col.startswith('dsc')])
        df['Method'] = [method] * df.shape[0]
        df_all.append(df)
    df = pd.concat(df_all)
    return df, methods, num_classes+1  # +1 for background

if __name__ == '__main__':
    run_make_summary()