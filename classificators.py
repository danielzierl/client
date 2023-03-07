
import numpy as np
from xgboost import XGBClassifier
import lightgbm
from sklearn.kernel_ridge import KernelRidge
from nnm import load_data
import torch


def make_classifiers():
    data_dir = "./data/new/final_data"
    # data_dir = "data/new/my_data/data1.csv"

    X_train, X_val, y_train, y_val = load_data(data_dir, 0.2, dir=True)

    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
                  gamma=0, gpu_id=-1, importance_type='gain',
                  interaction_constraints='', learning_rate=0.300000012,
                  max_delta_step=0, max_depth=10, min_child_weight=1,
                  monotone_constraints='()', n_estimators=1000, n_jobs=20,
                  num_parallel_tree=1, objective='multi:softprob', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
                  tree_method='exact',
                  validate_parameters=1, verbosity=None)

    lgm = lightgbm.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                importance_type='split', learning_rate=0.1, max_depth=-1,
                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                n_estimators=1000, n_jobs=-1, num_leaves=31, objective=None,
                random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    # KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


    models = [xgb, lgm]
    for model in models:
        model.fit(X_train, y_train)


        y_pred = model.predict(X_val)
        # print(y_pred)
        # accuracy = accuracy_score(y_val, y_pred)
        #
        # print(accuracy)
    return models

class wrapTorch:
    def __init__(self, torch_model, device):
        self.torch_model = torch_model
        self.device = device

    def predict(self, xarr):
        preds = self.torch_model(torch.from_numpy(xarr).to(torch.float32).to(self.device))
        return [int(preds.argmax())]

def combine_models():
    save_path = "saved_model"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from nnm import torch_model
    torch_model.to(device)
    torch_model.load_state_dict(torch.load(save_path))
    torch_model = wrapTorch(torch_model, device)

    return [*make_classifiers()]


if __name__ == "__main__":
    make_classifiers()