from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


MODEL_CONFIG = {
    "xgboost_multiclass": {
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
    "random_forest_clf": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 2, 4],
        }
    },
    "xgboost_regressor": {
        "estimator": XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
        ),
        "param_grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
        }
    },
    "random_forest_regressor": {
        "estimator": RandomForestRegressor(random_state=42),
        "param_grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 2, 4],
        }
    },
}
