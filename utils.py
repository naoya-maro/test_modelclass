import datetime
import os
import pickle
from typing import Optional, List, Dict, Any, Type
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import yaml
from model import ModelInterface

def japanese_font_setup(font_path: str = "/home/naoya/font/BIZ_UDGothic/BIZUDGothic-Regular.ttf"):
    path = Path(font_path)
    assert path.exists(), f"{path.resolve()} のファイルは見つかりません。"
    fontManager.addfont(path)
    plt.rcParams["font.family"] = (FontProperties(fname=str(path)).get_name())

# YAMLファイルに保存関数
def save_param(model, best_params):
    filename = f'Params/{model}/{model}_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
    with open(filename, 'w') as yaml_file:
        yaml.dump(best_params, yaml_file, default_flow_style=False)

#YAMLファイルから取り出し
def open_param(filename) -> dict:
    current_dir = Path(__file__).resolve().parent
    filename = os.path.join(current_dir, filename)
    with open(filename, 'r', encoding="utf-8") as yaml_file:
        loaded_params = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return loaded_params

#resultsフォルダの作成
def save_dir_func(Dir):
    current_dir_name = os.path.basename(os.getcwd())
    #保存先のベースフォルダを作成 (results/カレントディレクトリ名)
    save_dir = os.path.join(Dir+"/results", current_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

#モデルの保存
def model_save(model, fold_idx, save_dir, trial=None):
    if trial is not None:
        model_filename = f"fold_{fold_idx}_trial_{trial.number}_model.pkl"
    else:
        model_filename = f"fold_{fold_idx}_model.pkl"
    save_path = os.path.join(save_dir, model_filename)
    
    # モデルをpickleで保存
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"エラー: モデルの保存に失敗しました - {e}")


def training_pipeline(
    model: ModelInterface, X_train, y_train, X_val, y_val=None, **kwargs
    ) -> tuple[ModelInterface, np.ndarray | None]:
    '''
    モデルの学習と予測を行うパイプライン関数
    '''
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)
    oof = model.predict(X_val)
    return model, oof

def run_cv(X, y, folds, model_factory, **kwargs):
    """
    全体のCVループを管理する
    """
    oof_preds = np.zeros(len(X))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"--- Fold {fold} start ---")
        model = model_factory()
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model_fitted, oof = training_pipeline(
            model, X_train, y_train, X_val, y_val=y_val, **kwargs
        )

        oof_preds[val_idx] = oof
        models.append(model_fitted)
        
    return oof_preds, models

def submittion(X_test, sample_dir, save_dir):
    submit = pd.read_csv(sample_dir, encoding='shift-jis', header=None)
    submit[1] = X_test
    submit.to_csv(save_dir, index=False, header=False)
    print(f"Submit file saved to: {save_dir}")
    return submit



class CrossValidator:
    """
    クロスバリデーションクラス
    
    Usage:
        cv = CrossValidator(
            model_class=LightGBMModel,
            model_params=params,
            n_splits=5,
            stratified=False,
        )
        oof_preds, models = cv.fit(X, y)
    """

    def __init__(
        self,
        model_class: Type[ModelInterface],
        model_params: dict,
        metric_func: Optional[callable] = None,
    ):
        """
        Args:
            model_class   : ModelInterface を継承したモデルクラス（インスタンスではなくクラス自体）
            model_params  : モデルのハイパーパラメータ辞書
            n_splits      : fold 数
            stratified    : StratifiedKFold を使う場合 True（分類タスク向け）
            shuffle       : データをシャッフルするか
            random_state  : 乱数シード
            metric_func   : スコア計算関数 metric_func(y_true, y_pred) -> float
                            None の場合はスコア計算をスキップ
        """
        self.model_class = model_class
        self.model_params = model_params
        self.metric_func = metric_func

        self.models: List[ModelInterface] = []
        self.oof_preds: Optional[np.ndarray] = None
        self.scores: List[float] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> tuple[np.ndarray, List[ModelInterface]]:
        """
        クロスバリデーションを実行する。

        Args:
            X           : 特徴量
            y           : ターゲット
            fit_kwargs  : model.fit() に渡す追加引数
                          例: {'weight_train': w, 'stopping_rounds': 50}
                          weight_train / weight_val がある場合、fold ごとに自動で分割される。

        Returns:
            oof_preds : Out-of-Fold 予測値 (shape: [n_samples])
            models    : 各 fold で学習したモデルのリスト
        """
        X = np.array(X) if isinstance(X, pd.DataFrame) else X
        y = np.array(y) if isinstance(y, pd.Series) else y

        oof_preds = np.zeros(len(y))
        self.models = []
        self.scores = []

        for fold, (train_idx, val_idx) in enumerate(folds):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # モデル生成・学習
            model = self.model_class(self.model_params.copy())
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)

            # OOF 予測
            oof_preds[val_idx] = model.predict(X_val)
            self.models.append(model)

            # スコア計算
            if self.metric_func is not None:
                score = self.metric_func(y_val, oof_preds[val_idx])
                self.scores.append(score)
                print(f"  Fold {fold + 1} Score: {score:.4f}")

        self.oof_preds = oof_preds

        if self.scores:
            print(f"\n{'='*40}")
            print(f"  CV Score: {np.mean(self.scores):.4f} ± {np.std(self.scores):.4f}")
            print(f"{'='*40}\n")

        return oof_preds, self.models

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        全 fold のモデルで予測し、平均を返す
        """
        if not self.models:
            raise RuntimeError("fit() を先に実行してください。")
        X = np.array(X) if isinstance(X, pd.DataFrame) else X
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)

    @property
    def cv_score(self) -> Optional[float]:
        return float(np.mean(self.scores)) if self.scores else None