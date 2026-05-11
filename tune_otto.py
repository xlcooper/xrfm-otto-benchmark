import sys, json

from src.preprocessing import load_otto_data, split_data
from configs.config import MODEL_CONFIG

from pathlib import Path
from sklearn.model_selection import GridSearchCV



def run_tune(model_name, X_train, y_train):

    # choose current model config
    model_config = MODEL_CONFIG[model_name]

    # model fitting/training by GridSearchCV
    grid_search = GridSearchCV(
        estimator=model_config["estimator"],
        param_grid=model_config["param_grid"],
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,  # M芯片多核处理
        verbose=1  
    )

    grid_search.fit(X_train, y_train)

    # get params
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_


    return best_params, best_cv_score

def main(model_name):
    # load data
    X, y, _ = load_otto_data()

    # split dataset
    X_train, _, y_train, _ = split_data(X ,y)
    best_params, best_cv_score = run_tune(model_name, X_train, y_train)

    Path("configs").mkdir(exist_ok=True)        # 防御
    with open(f"configs/best_{model_name}.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_cv_score": float(best_cv_score),
        }, f, indent=4)

if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)