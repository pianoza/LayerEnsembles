import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
import pickle
from sklearn.calibration import calibration_curve
import itertools

sys.path.append('/home/kaisar/EuCanImage/Coding/UQComparison/new-uncertainty-estimation-method')
import configs
from utils import make_folders
from exp_list import get_experiments_list

dataset = configs.DATASET
experiments = get_experiments_list(dataset)
if dataset == 'inbreast':
    calib_plot_out_name = 'Inbreast-calibration_plot.png'
    calib_plot_title = 'Calibration plot'
elif dataset == 'bcdr':
    calib_plot_out_name = 'BCDR-calibration_plot.png'
    calib_plot_title = 'Calibration plot'
else:
    raise ValueError(f"{dataset} not implemented")

# load results per experiment
methods = list()
y_true = dict()
y_prob = dict()
fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot([0, 1], [0, 1], 'k--')
for method, exp_name in experiments.items():
    methods.append(method)
    models_path, figures_path, seg_out_path = make_folders(configs.RESULTS_PATH, exp_name)
    print(method)
    y_true[method] = []
    y_prob[method] = []
    probas_path = f'results_{exp_name}-calibration_pairs.pkl'
    # calibration pairs is a list of tuples of (lbl, probs)
    calibration_pairs = pickle.load( open(str(seg_out_path / probas_path), "rb"))
    y_true[method].extend([a[0] for a in calibration_pairs])
    y_prob[method].extend([a[1] for a in calibration_pairs])
    yt = list(itertools.chain.from_iterable([x.flatten() for x in y_true[method]]))
    yp = list(itertools.chain.from_iterable([x.flatten() for x in y_prob[method]]))
    prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10)
    ax.plot(prob_pred, prob_true, label=method)
plt.title(calib_plot_title)
plt.xlabel('Predicted probabilities')
plt.ylabel('Observed probabilities')
plt.legend()
figures_save_path = configs.RESULTS_PATH / 'summary'
figures_save_path.mkdir(parents=True, exist_ok=True)
plt.savefig(str(figures_save_path / calib_plot_out_name))