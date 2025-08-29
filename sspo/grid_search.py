import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
import optuna
from sspo.data import ACCESS_TOKEN, SEGMENT_LIST, collect_data
from sspo.model import save_xgb_reg
from stravalib import Client as StravaClient


def objective(trial: optuna.trial._trial.Trial) -> float:
    params = {'n_estimators':trial.suggest_int('n_estimators', 500, 2000),
              'max_depth':trial.suggest_int('max_depth', 3, 11),
              'max_leaves':trial.suggest_int('max_leaves', 0, 5),
              'grow_policy':trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
              'learning_rate':trial.suggest_float('learning_rate', 0.005, 0.01, log=True),
              'n_jobs':-1,
              'subsample':trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
              'colsample_bylevel':trial.suggest_categorical('colsample_bylevel', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
              'reg_alpha':trial.suggest_float('reg_alpha', 1e-3, 100, log=True),
              'reg_lambda':trial.suggest_float('reg_lambda', 1e-3, 100, log=True),
              'seed':42
              }
    xgb_reg = XGBRegressor(**params, eval_metric=["rmse"])
    xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    rmse = round(mean_squared_error(y_test.values, xgb_reg.predict(X_test))**0.5, 2)

    return rmse

df = collect_data(StravaClient(access_token=ACCESS_TOKEN), SEGMENT_LIST)
X = df.drop(columns=['time'])
y = df.time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

study = optuna.create_study()
study.optimize(objective, n_trials=100)
best_xgb_reg = XGBRegressor(**study.best_params, eval_metric=["rmse"], seed=42)
best_xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
rmse = round(mean_squared_error(y_test.values, best_xgb_reg.predict(X_test))**0.5, 2)
save_xgb_reg(best_xgb_reg, f"xgb_reg_{str(rmse).replace(".", "_")}.pkl")
