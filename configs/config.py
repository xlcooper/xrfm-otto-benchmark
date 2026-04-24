from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


MODEL_CONFIG = {
    "decision_tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth": [15, 20, 25, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf" : [1, 2, 4, 8],
        }
    },
    "xgboost": {
        "estimator": XGBClassifier(
            objective="multi:softprob",     
            eval_metric="mlogloss",
            random_state=42,
        ),
        "param_grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        }
    },
    # new model
}
