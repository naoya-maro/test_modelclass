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


class RobustNumericalScaler(BaseEstimator, TransformerMixin):
    """
    CONFIG["transform_steps"] に指定された数値変換のシーケンスを適用します。
    対応している変換：'median_center'、'robust_scale'、'smooth_clip'、'l2_normalize'。
    'one_hot' および 'embedding' は認識されますが、スキップされます。
    """

    def __init__(self, transform_steps):
        self._transform_steps = [t for t in transform_steps
                      if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X: np.ndarray, y=None):
        if "median_center" in self._transform_steps or "robust_scale" in self._transform_steps:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = X.copy().astype(np.float32)
        for tfm in self._transform_steps:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


class FoldPreprocessor():
    """Fold 前処理クラス: スケーリング(Standard/Robust) + group 情報のオーディナルエンコーディング"""
    def __init__(self, scaler_type: Optional[str] = 'standard', transform_steps: Optional[list] = None, pca_components: Optional[int] = None):
        self.scaler_type = str(scaler_type).lower().strip() if scaler_type is not None else 'none'
        self.group_map = None
        self.pca_components = pca_components
        if self.pca_components is not None and self.pca_components > 0:
            self.pca = PCA(n_components=self.pca_components, random_state=42)

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustNumericalScaler(transform_steps)
        else:
            self.scaler = None

    def _encode_groups(self, groups, fit=False):
        """group情報をオーディナルエンコード"""
        if groups is None:
            return None
        
        if fit:
            unique_groups = list(dict.fromkeys(groups))
            self.group_map = {g: i for i, g in enumerate(unique_groups)}
        
        if self.group_map is None:
            raise ValueError("fit() を先に実行してください。")
        
        # 未知のグループに対応：存在しないグループは-1にマップ
        return np.array([self.group_map.get(g, -1) for g in groups])

    def fit(self, X, groups=None):
        if self.scaler is not None:
            X = np.asarray(X)
            self.scaler.fit(X)
        if self.pca_components is not None and self.pca_components > 0:
            self.pca.fit(X)
        if groups is not None:
            self._encode_groups(groups, fit=True)
        return self

    def transform(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(np.asarray(X))
        if self.pca_components is not None and self.pca_components > 0:
            X = self.pca.transform(X)
        return X

    def encode_groups(self, groups):
        """group_ids が必要な場合は別途呼び出す"""
        return self._encode_groups(groups)



class CrossValidator:
    """
    クロスバリデーションクラス
    """
    def __init__(
        self,
        model_class: Type[ModelInterface],
        config: dict,
        metric_func: Optional[callable] = None,
    ):
        """
        Args:
            model_class   : ModelInterface を継承したモデルクラス（インスタンスではなくクラス自体）
            config  : モデルのハイパーパラメータ辞書
            n_splits      : fold 数
            stratified    : StratifiedKFold を使う場合 True（分類タスク向け）
            shuffle       : データをシャッフルするか
            random_state  : 乱数シード
            metric_func   : スコア計算関数 metric_func(y_true, y_pred) -> float
                            None の場合はスコア計算をスキップ
        """
        self.model_class = model_class
        self.config = config
        self.cv_cfg = config['CV_params']
        self.metric_func = metric_func

        self.models: List[ModelInterface] = []
        self.oof_preds: Optional[np.ndarray] = None
        self.scores: List[float] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        groups: Optional[np.ndarray | pd.Series] = None,
        folds: list[tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> tuple[np.ndarray, List[ModelInterface]]:
        """
        クロスバリデーションを実行する。

        Args:
            X           : 特徴量
            y           : ターゲット
            groups      : グループ情報（fold_preprocess_func に渡される）
            folds       : fold のインデックス
            fold_preprocess_func : 前処理クラスまたはインスタンス
            fit_kwargs  : model.fit() に渡す追加引数
                          例: {'weight_train': w, 'stopping_rounds': 50}
                          weight_train / weight_val がある場合、fold ごとに自動で分割される。

        Returns:
            oof_preds : Out-of-Fold 予測値 (shape: [n_samples])
            models    : 各 fold で学習したモデルのリスト
        """
        X = np.array(X) if isinstance(X, pd.DataFrame) else X
        y = np.array(y) if isinstance(y, pd.Series) else y
        if groups is not None:
            groups = np.array(groups) if isinstance(groups, pd.Series) else groups

        oof_preds = np.zeros(len(y))
        self.models = []
        self.scores = []
        self.preprocessors = []
        if self.config.get('Fit_params'):
            self.kwargs = {k: v for k, v in self.config['Fit_params'].items()}
        else:
            self.kwargs = {}

        for fold, (train_idx, val_idx) in enumerate(folds):
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            if groups is not None:
                groups_train = groups[train_idx]
            
            # Fold内での前処理
            preprocessor = FoldPreprocessor(**self.cv_cfg)
            if groups is not None:
                preprocessor.fit(X_train, groups_train)
                X_train = preprocessor.transform(X_train)
                X_val = preprocessor.transform(X_val)
                group_ids_train  = preprocessor.encode_groups(groups_train)
                self.kwargs['group_ids'] = group_ids_train
            else:
                preprocessor.fit(X_train)
                X_train = preprocessor.transform(X_train)
                X_val = preprocessor.transform(X_val)
            self.preprocessors.append(preprocessor)

            # モデル生成・学習
            model = self.model_class(self.config.copy())
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **self.kwargs)

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
        
        preds = []
        for i, model in enumerate(self.models):
            # avoid mutating input across iterations
            X_input = np.array(X) if isinstance(X, pd.DataFrame) else X.copy()
            if self.preprocessors:
                X_input = self.preprocessors[i].transform(X_input)
            preds.append(model.predict(X_input))
        return np.array(preds).mean(axis=0)
    
    def predict_with_fold(self, X_list: list[np.ndarray | pd.DataFrame]) -> np.ndarray:
        """
        各 fold に対応する個別の入力データ X を受け取り、それぞれのモデルで予測して平均を返す
        Args:
            X_list: list[X_fold0, X_fold1, ..., X_foldN]
        """
        if not self.models:
            raise RuntimeError("fit() を先に実行してください。")
        
        if len(X_list) != len(self.models):
            raise ValueError(f"入力データの数({len(X_list)})がモデルの数({len(self.models)})と一致しません。")

        preds = []
        for i, model in enumerate(self.models):
            # 個別の X を取得
            X = X_list[i]
            X_input = np.array(X) if isinstance(X, pd.DataFrame) else X.copy()
            
            if self.preprocessors:
                X_input = self.preprocessors[i].transform(X_input)
                
            # その Fold のモデルで予測
            preds.append(model.predict(X))
        
        # 全モデルの予測値を平均 (アンサンブル)
        return np.array(preds).mean(axis=0)

    @property
    def cv_score(self) -> Optional[float]:
        return float(np.mean(self.scores)) if self.scores else None
    

def save_deep_features(model, loader, save_path, prefix="", weights_path=None):
    """モデルの特徴抽出部分を使って、全データの特徴量を保存する関数"""
    all_features = []
    all_targets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 学習済み重みを読み込む
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets, groups in loader:
            inputs = inputs.to(device)
            # 32次元ベクトルを取得
            feat = model.extract_features(inputs)
            all_features.append(feat.cpu().numpy())
            all_targets.append(targets.numpy())
            
    # 全データを結合
    features_array = np.vstack(all_features)
    targets_array = np.concatenate(all_targets)
    
    # 保存
    np.save(f"{save_path}/{prefix}_feat.npy", features_array)
    np.save(f"{save_path}/{prefix}_target.npy", targets_array)
    return features_array, targets_array
