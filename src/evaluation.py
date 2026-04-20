from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_auc_score

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
    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class='ovr',
        average='macro'
    )
    return auc