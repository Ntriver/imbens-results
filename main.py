from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
from imbalanced_ensemble.datasets import generate_imbalance_data
from imbalanced_ensemble.utils import evaluate_print
from imbalanced_ensemble.visualizer import ImbalancedEnsembleVisualizer
from imbalanced_ensemble.metrics import geometric_mean_score
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score, roc_auc_score, precision_recall_curve, auc

from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
import os
import numpy as np

from compute_metrics import compute_metrics
from fit_ensemble_with_gridsearch import fit_ensemble_with_gridsearch

# X_train, X_test, y_train, y_test = generate_imbalance_data(n_samples=200, weights=[.9,.1], test_size=.5,random_state=2025)
# X, y = datasets.make_moons(noise=0.352, random_state=2025, n_samples=500)
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2025)

ensemble_list = [
                # 'SelfPacedEnsembleClassifier',
                #  'BalanceCascadeClassifier',
                #  'BalancedRandomForestClassifier',
                #  'EasyEnsembleClassifier',
                #  'RUSBoostClassifier',
                #  'UnderBaggingClassifier',
                #  'OverBoostClassifier',
                #  'SMOTEBoostClassifier',
                 # 'KmeansSMOTEBoostClassifier',# not working well
                 # 'OverBaggingClassifier',
                 'SMOTEBaggingClassifier',
                 'AdaCostClassifier',
                 'AdaUBoostClassifier',
                 'AsymBoostClassifier',
                 'CompatibleAdaBoostClassifier',
                 'CompatibleBaggingClassifier'
                 ]
n_estimator_list = [10,20,30,40,50]#

dataset_root = './data/10T5CV/'
all_file_names = os.listdir(dataset_root)

all_res = []
for ensemble_name in ensemble_list:
    for file_name in all_file_names:####
        res = []
        for i in range(50):#10 times 5 cv
            print(f'{ensemble_name}, {file_name} fold # {i}')
            npzfile = np.load(dataset_root+file_name+'/'+file_name+'-'+str(i)+'.npz')
            x_train, x_test, y_train, y_test = npzfile['x_train'], npzfile['x_test'], npzfile['y_train'], npzfile['y_test']
            res.append(fit_ensemble_with_gridsearch(x_train,y_train,x_test,y_test,ensemble_name,n_estimator_list))


        saved_path = './results/' + file_name + '/'
        if not os.path.isdir(saved_path):
            os.mkdir(saved_path)
        np.save(saved_path + ensemble_name + '.npy', np.array(res))
        # exit()





