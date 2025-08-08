
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score, roc_curve, precision_recall_curve, auc,average_precision_score,roc_auc_score
from sklearn.metrics import precision_score, recall_score


def compute_metrics(y_true, y_pred, y_score):
    """
    pos_label = +1, neg_label = -1
    :param y_true:
    :param y_pred:
    :param y_score:
    :return:
    """
    f1 = f1_score(y_true, y_pred)
    mc = matthews_corrcoef(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true,y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score[:,1],pos_label=1)
    auroc = auc(fpr, tpr)
    # auroc2 = roc_auc_score(y_true,y_score[:,1])
    # print(f"auroc: {auroc}, auroc2: {auroc2}")
    precision, recall, _ = precision_recall_curve(y_true, y_score[:,1], pos_label=1)
    auprc = auc(recall, precision)
    # auprc2 = average_precision_score(y_true, y_score[:,1])
    # print(f"auprc: {auprc}, auprc2: {auprc2}")

    p_pos = precision_score(y_true, y_pred,pos_label=1)
    r_pos = recall_score(y_true, y_pred,pos_label=1)
    p_neg = precision_score(y_true, y_pred,pos_label=-1)
    r_neg = recall_score(y_true, y_pred,pos_label=-1)

    return [acc, ba, f1, mc, auroc, auprc, p_pos, r_pos, p_neg, r_neg]
