import sys
from datetime import datetime

from src.preprocessing import load_data, split_data
from src.evaluation import get_classification_report, get_f1_score, get_accuracy, get_confusion_matrix, get_auc_roc
from src.reporting import save_result_json, save_result_md
from src.feature_analysis import run_permutation_importance
from config import MODEL_CONFIG

from sklearn.model_selection import GridSearchCV



def main(model_name):
    # load data
    X, y, le = load_data()
    class_names = le.classes_

    # split dataset
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X ,y)

    # model selection
    config = MODEL_CONFIG[model_name]

    # model fitting/training
    grid_search = GridSearchCV(
        config["estimator"],
        param_grid=config["param_grid"],
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1  
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # predict on val
    y_pred = best_model.predict(X_val)
    y_pred_train = best_model.predict(X_train)
    y_prob = best_model.predict_proba(X_val)


    # evaluation
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    val_f1_score = get_f1_score(y_true=y_val, y_pred=y_pred)
    val_accuracy = get_accuracy(y_true=y_val, y_pred=y_pred)
    train_accuracy = get_accuracy(y_true=y_train, y_pred=y_pred_train)
    val_auc_roc = get_auc_roc(y_true=y_val, y_prob=y_prob)
    confusion_matrix = get_confusion_matrix(y_true=y_val, y_pred=y_pred)
    report = get_classification_report(y_true=y_val, y_pred=y_pred, class_names=class_names)

    result = {
        "model": model_name,
        "best_params": best_params,
        "best_cv_score": float(best_cv_score),
        "val_f1_score": float(val_f1_score),
        "val_accuracy": float(val_accuracy),
        "val_auc_roc": float(val_auc_roc),
        "train_accuracy": float(train_accuracy),
        "confusion_matrix": confusion_matrix.tolist(),
        "report": report
    }

    print()
    print(confusion_matrix)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_result_json(result=result, model_name=model_name, timestamp=timestamp)
    save_result_md(result=result, model_name=model_name, timestamp=timestamp)

    feature_names = X.columns.tolist()
    run_permutation_importance(model=best_model, X_val=X_val, y_val=y_val, feature_names=feature_names, model_name=model_name)

if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)