import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

test = 'ttest_rel'
test = 'wilcoxon'
if test == 'wilcoxon':
    plain = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/BCDR-Plain-Aug/segmentations/results_BCDR-Plain-Aug.csv')
    de = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/DE-BCDR-GoldStandard-Aug/segmentations/results_DE-BCDR-GoldStandard-Aug.csv')
    le = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/BCDR-TestRun-3-T7/segmentations/results_BCDR-TestRun-3-T7.csv')
    # BCDR LE vs DE
    # DSC
    stat, pval = wilcoxon(le['dsc_norm'], de['dsc_norm'], zero_method='wilcox', alternative='two-sided')
    print(f'BCDR LE vs DE (DSC): {stat}, {pval}')
    # MHD
    stat, pval = wilcoxon(le['mhd'], de['mhd'], zero_method='wilcox', alternative='two-sided')
    print(f'BCDR LE vs DE (MHD): {stat}, {pval}')
    # NLL
    stat, pval = wilcoxon(le['nll'], de['nll'], zero_method='wilcox', alternative='two-sided')
    print(f'BCDR LE vs DE (NLL): {stat}, {pval}')

    # BCDR LE vs Plain
    # DSC
    stat, pval = wilcoxon(le['dsc_norm'], plain['dsc_norm'], zero_method='wilcox', alternative='two-sided')
    print(f'BCDR LE vs Plain (DSC): {stat}, {pval}')
    # MHD
    stat, pval = wilcoxon(le['mhd'], plain['mhd'], zero_method='wilcox', alternative='two-sided')
    print(f'BCDR LE vs Plain (MHD): {stat}, {pval}')
    # NLL
    stat, pval = wilcoxon(le['nll'], plain['nll'], zero_method='wilcox', alternative='two-sided')
    print(f'BCDR LE vs Plain (NLL): {stat}, {pval}')


    plain = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/MnM-Plain-CE/segmentations/results_MnM-Plain-CE.csv')
    de = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/MnM-DE-GoldStandard-CE/segmentations/results_MnM-DE-GoldStandard-CE.csv')
    le = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/MnM-TestRun-4-T5-inPaper/segmentations/results_MnM-TestRun-4-T5-inPaper.csv')

    # MnM LE vs DE
    # DSC
    stat, pval = wilcoxon(le['dsc_norm'], de['dsc_norm'], zero_method='wilcox', alternative='two-sided')
    print(f'MnM LE vs DE (DSC): {stat}, {pval}')
    # MHD
    stat, pval = wilcoxon(le['mhd'], de['mhd'], zero_method='wilcox', alternative='two-sided')
    print(f'MnM LE vs DE (MHD): {stat}, {pval}')
    # NLL
    stat, pval = wilcoxon(le['nll'], de['nll'], zero_method='wilcox', alternative='two-sided')
    print(f'MnM LE vs DE (NLL): {stat}, {pval}')

    # MnM LE vs Plain
    # DSC
    stat, pval = wilcoxon(le['dsc_norm'], plain['dsc_norm'], zero_method='wilcox', alternative='two-sided')
    print(f'MnM LE vs Plain (DSC): {stat}, {pval}')
    # MHD
    stat, pval = wilcoxon(le['mhd'], plain['mhd'], zero_method='wilcox', alternative='two-sided')
    print(f'MnM LE vs Plain (MHD): {stat}, {pval}')
    # NLL
    stat, pval = wilcoxon(le['nll'], plain['nll'], zero_method='wilcox', alternative='two-sided')
    print(f'MnM LE vs Plain (NLL): {stat}, {pval}')

    # MnM Multi-Class
    # LE vs DE
    classes = ['LV', 'MYO', 'RV']
    for cl in range(1, 4):
        # DSC
        stat, pval = wilcoxon(le[f'dsc_cl{cl}'], de[f'dsc_cl{cl}'], zero_method='wilcox', alternative='two-sided')
        print(f'MnM Multi-Class LE vs DE (DSC, {classes[cl-1]}): {stat}, {pval}')
        # MHD
        stat, pval = wilcoxon(le[f'mhd_cl{cl}'], de[f'mhd_cl{cl}'], zero_method='wilcox', alternative='two-sided')
        print(f'MnM Multi-Class LE vs DE (MHD, {classes[cl-1]}): {stat}, {pval}')

    # LE vs Plain
    for cl in range(1, 4):
        # DSC
        stat, pval = wilcoxon(le[f'dsc_cl{cl}'], plain[f'dsc_cl{cl}'], zero_method='wilcox', alternative='two-sided')
        print(f'MnM Multi-Class LE vs Plain (DSC, {classes[cl-1]}): {stat}, {pval}')
        # MHD
        stat, pval = wilcoxon(le[f'mhd_cl{cl}'], plain[f'mhd_cl{cl}'], zero_method='wilcox', alternative='two-sided')
        print(f'MnM Multi-Class LE vs Plain (MHD, {classes[cl-1]}): {stat}, {pval}')
elif test == 'ttest_rel':
    plain = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/BCDR-Plain-Aug/segmentations/results_BCDR-Plain-Aug.csv')
    de = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/DE-BCDR-GoldStandard-Aug/segmentations/results_DE-BCDR-GoldStandard-Aug.csv')
    le = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/BCDR-TestRun-3-T7/segmentations/results_BCDR-TestRun-3-T7.csv')
    # BCDR LE vs DE
    # DSC
    stat, pval = ttest_rel(le['dsc_norm'], de['dsc_norm'], alternative='two-sided')
    print(f'BCDR LE vs DE (DSC): {stat}, {pval}')
    # MHD
    stat, pval = ttest_rel(le['mhd'], de['mhd'], alternative='two-sided')
    print(f'BCDR LE vs DE (MHD): {stat}, {pval}')
    # NLL
    stat, pval = ttest_rel(le['nll'], de['nll'], alternative='two-sided')
    print(f'BCDR LE vs DE (NLL): {stat}, {pval}')

    # BCDR LE vs Plain
    # DSC
    stat, pval = ttest_rel(le['dsc_norm'], plain['dsc_norm'], alternative='two-sided')
    print(f'BCDR LE vs Plain (DSC): {stat}, {pval}')
    # MHD
    stat, pval = ttest_rel(le['mhd'], plain['mhd'], alternative='two-sided')
    print(f'BCDR LE vs Plain (MHD): {stat}, {pval}')
    # NLL
    stat, pval = ttest_rel(le['nll'], plain['nll'], alternative='two-sided')
    print(f'BCDR LE vs Plain (NLL): {stat}, {pval}')

    plain = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/MnM-Plain-CE/segmentations/results_MnM-Plain-CE.csv')
    de = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/MnM-DE-GoldStandard-CE/segmentations/results_MnM-DE-GoldStandard-CE.csv')
    le = pd.read_csv('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method/results/MnM-TestRun-4-T5-inPaper/segmentations/results_MnM-TestRun-4-T5-inPaper.csv')

    # MnM LE vs DE
    # DSC
    stat, pval = ttest_rel(le['dsc_norm'], de['dsc_norm'], alternative='two-sided')
    print(f'MnM LE vs DE (DSC): {stat}, {pval}')
    # MHD
    stat, pval = ttest_rel(le['mhd'], de['mhd'], alternative='two-sided')
    print(f'MnM LE vs DE (MHD): {stat}, {pval}')
    # NLL
    stat, pval = ttest_rel(le['nll'], de['nll'], alternative='two-sided')
    print(f'MnM LE vs DE (NLL): {stat}, {pval}')

    # MnM LE vs Plain
    # DSC
    stat, pval = ttest_rel(le['dsc_norm'], plain['dsc_norm'], alternative='two-sided')
    print(f'MnM LE vs Plain (DSC): {stat}, {pval}')
    # MHD
    stat, pval = ttest_rel(le['mhd'], plain['mhd'], alternative='two-sided')
    print(f'MnM LE vs Plain (MHD): {stat}, {pval}')
    # NLL
    stat, pval = ttest_rel(le['nll'], plain['nll'], alternative='two-sided')
    print(f'MnM LE vs Plain (NLL): {stat}, {pval}')

    # MnM Multi-Class
    # LE vs DE
    classes = ['LV', 'MYO', 'RV']
    for cl in range(1, 4):
        # DSC
        stat, pval = ttest_rel(le[f'dsc_cl{cl}'], de[f'dsc_cl{cl}'], alternative='two-sided')
        print(f'MnM Multi-Class LE vs DE (DSC, {classes[cl-1]}): {stat}, {pval}')
        # MHD
        stat, pval = ttest_rel(le[f'mhd_cl{cl}'], de[f'mhd_cl{cl}'], alternative='two-sided')
        print(f'MnM Multi-Class LE vs DE (MHD, {classes[cl-1]}): {stat}, {pval}')

    # LE vs Plain
    for cl in range(1, 4):
        # DSC
        stat, pval = ttest_rel(le[f'dsc_cl{cl}'], plain[f'dsc_cl{cl}'], alternative='two-sided')
        print(f'MnM Multi-Class LE vs Plain (DSC, {classes[cl-1]}): {stat}, {pval}')
        # MHD
        stat, pval = ttest_rel(le[f'mhd_cl{cl}'], plain[f'mhd_cl{cl}'], alternative='two-sided')
        print(f'MnM Multi-Class LE vs Plain (MHD, {classes[cl-1]}): {stat}, {pval}')