import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
import optuna
from sspo.data import ACCESS_TOKEN, SEGMENT_LIST, collect_data
from sspo.model import save_xgb_reg, load_xgb_reg
from stravalib import Client as StravaClient


def objective(trial: optuna.trial._trial.Trial, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
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


def optimize_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[XGBRegressor, float]:
    study = optuna.create_study()
    study.optimize(objective(X_train, X_test, y_train, y_test), n_trials=100)
    new_xgb_reg = XGBRegressor(**study.best_params, eval_metric=["rmse"], seed=42)
    new_xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    new_rmse = round(mean_squared_error(y_test.values, new_xgb_reg.predict(X_test))**0.5, 2)

    return (new_xgb_reg, new_rmse)


def get_best_model_path() -> str:
    directory = os.listdir("/")
    for file in directory:
        metric = 0
        if file.startswith("xgb_reg_"):
            splitted = file.split("_")
            new_metric = splitted[2] + splitted[2]/100
            if new_metric < metric:
                xgb_path = file
                metric = new_metric

    return xgb_path

def 
    current_xgb_reg = load_xgb_reg(xgb_path)
    current_xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    current_rmse = round(mean_squared_error(y_test.values, current_xgb_reg.predict(X_test))**0.5, 2)
    if new_rmse < current_rmse:
        save_xgb_reg(new_xgb_reg, f"xgb_reg_{str(new_rmse).replace(".", "_")}.pkl")



def main():
    #NEED TO BE REPLACED WITH THE CLOUD DATABASE
    file_path = Path("database/current/10475048_data.parquet")
    df = pd.read_parquet(file_path)
    df.set_index("id", inplace=True)
    X = df.drop(columns=['time'])
    y = df.time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_xgb_reg, new_rmse = optimize_model(X_train, X_test, y_train, y_test)

    current_xgb_reg = load_xgb_reg(get_best_model_path())
    current_xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    current_rmse = round(mean_squared_error(y_test.values, current_xgb_reg.predict(X_test))**0.5, 2)
    if new_rmse < current_rmse:
        save_xgb_reg(new_xgb_reg, f"xgb_reg_{str(new_rmse).replace(".", "_")}.pkl")
