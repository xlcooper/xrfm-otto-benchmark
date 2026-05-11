from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
)

def get_classification_report(y_true, y_pred, class_names):
    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
    )
    return report

def get_f1_score(y_true, y_pred):
    f1 = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average='macro'
        )
    return f1


def get_accuracy(y_true, y_pred, normalize=True):
    accuracy = accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
        normalize=normalize,
    )
    return accuracy

def get_confusion_matrix(y_true, y_pred):
    C = confusion_matrix(
        y_true,
        y_pred)
    return C

def get_auc_roc(y_true, y_prob):
    if y_prob.ndim == 1:
        return roc_auc_score(y_true, y_prob)
    elif y_prob.shape[1] == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        return roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')


def get_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)


def get_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def get_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
