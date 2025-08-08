from imbalanced_ensemble.ensemble import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from compute_metrics import compute_metrics

def fit_ensemble_with_gridsearch(x_train,y_train,x_test,y_test,ensemble_estimator_name='SelfPacedEnsembleClassifier', n_estimator_list=[10,],random_state=2025):
    params = {'n_estimators': n_estimator_list}

    if ensemble_estimator_name=='SelfPacedEnsembleClassifier':
        ensemble_estimator = SelfPacedEnsembleClassifier(random_state=random_state)
    elif ensemble_estimator_name=='BalanceCascadeClassifier':
        ensemble_estimator = BalanceCascadeClassifier(random_state=random_state)
    elif ensemble_estimator_name=='BalancedRandomForestClassifier':
        ensemble_estimator = BalancedRandomForestClassifier(random_state=random_state)
    elif ensemble_estimator_name=='EasyEnsembleClassifier':
        ensemble_estimator = EasyEnsembleClassifier(random_state=random_state)
    elif ensemble_estimator_name=='RUSBoostClassifier':
        ensemble_estimator = RUSBoostClassifier(random_state=random_state)
    elif ensemble_estimator_name=='UnderBaggingClassifier':
        ensemble_estimator = UnderBaggingClassifier(random_state=random_state)
    elif ensemble_estimator_name=='OverBoostClassifier':
        ensemble_estimator = OverBoostClassifier(random_state=random_state)
    elif ensemble_estimator_name=='SMOTEBoostClassifier':
        ensemble_estimator = SMOTEBoostClassifier(random_state=random_state)
    elif ensemble_estimator_name=='KmeansSMOTEBoostClassifier':
        ensemble_estimator = KmeansSMOTEBoostClassifier(random_state=random_state,cluster_balance_threshold=0.1)
    elif ensemble_estimator_name=='OverBaggingClassifier':
        ensemble_estimator = OverBaggingClassifier(random_state=random_state)
    elif ensemble_estimator_name=='SMOTEBaggingClassifier':
        ensemble_estimator = SMOTEBaggingClassifier(random_state=random_state)
    elif ensemble_estimator_name=='AdaCostClassifier':
        ensemble_estimator = AdaCostClassifier(random_state=random_state)
    elif ensemble_estimator_name=='AdaUBoostClassifier':
        ensemble_estimator = AdaUBoostClassifier(random_state=random_state)
    elif ensemble_estimator_name=='AsymBoostClassifier':
        ensemble_estimator = AsymBoostClassifier(random_state=random_state)
    elif ensemble_estimator_name=='CompatibleAdaBoostClassifier':
        ensemble_estimator = CompatibleAdaBoostClassifier(random_state=random_state)
    elif ensemble_estimator_name=='CompatibleBaggingClassifier':
        ensemble_estimator = CompatibleBaggingClassifier(random_state=random_state)
    else:
        print('ensemble name not specified, using SelfPacedEnsemble as default.')
        ensemble_estimator = SelfPacedEnsembleClassifier(random_state=random_state)


    clf = GridSearchCV(estimator=ensemble_estimator, param_grid=params, scoring='balanced_accuracy',
                       cv=StratifiedKFold(n_splits=3))
    clf.fit(x_train, y_train)
    best_est = clf.best_estimator_
    y_pred = best_est.predict(x_test)
    y_score = best_est.predict_proba(x_test)
    return compute_metrics(y_test, y_pred, y_score)