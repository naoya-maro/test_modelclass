from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GroupKFold
import numpy as np
import pandas as pd

class TargetEncode(BaseEstimator, TransformerMixin):
    def __init__(self, categories='auto', noise_level=0, random_state=None,
                 cv=None, groups=None, agg_func='mean', smooth='auto'):
        self.categories = categories
        # self.k = k
        # self.f = f
        self.noise_level = noise_level
        self.encodings = dict()
        self.prior = dict()
        self.random_state = random_state
        self.cv = cv
        self.groups = groups
        self.agg_func = [agg_func] if isinstance(agg_func, str) else agg_func
        self.smooth = smooth

    def add_noise(self, series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))
    
    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("y cannot be None for TargetEncode.fit()")
        
        if isinstance(self.categories, str) and self.categories == 'auto':
            self.categories = X.select_dtypes(include=['object', 'category']).columns.tolist()
        elif isinstance(self.categories, str):
            self.categories = [self.categories]
        
        y_values = y.values if hasattr(y, 'values') else y
        
        for variable in self.categories:
            self.encodings[variable] = {}
            var_values = X[variable].values
            
            for agg_func in self.agg_func:
                prior = getattr(np, agg_func)(y_values)
                
                # groupbyの代わりにpd.DataFrameで直接集計
                df_agg = pd.DataFrame({'variable': var_values, 'target': y_values})
                stats = df_agg.groupby('variable', observed=True, sort=False)['target'].agg([agg_func, 'count', 'var'])
                if agg_func == 'std':
                        stats = stats.fillna(0)
                        
                m = self.smooth
                if self.smooth == 'auto':
                    # Empirical Bayes smoothing
                    stats['var'] = stats['var'].fillna(0)
                    variance_between = stats[agg_func].var()
                    avg_variance_within = stats['var'].mean()
                    if variance_between > 0:
                        m = avg_variance_within / variance_between
                    else:
                        m = 0  # No smoothing if no variance between groups
                encodings = (stats['count'].values * stats[agg_func].values + m * prior) / (stats['count'].values + m)
                
                #手動パラメータによる平滑化
                # smoothing = 1 / (1 + np.exp(-(stats['count'].values - self.k) / self.f))
                # encodings = prior * (1 - smoothing) + stats[agg_func].values * smoothing
                
                self.encodings[variable][agg_func] = dict(zip(stats.index, encodings))
                self.prior[agg_func] = prior
        return self
    
    def transform(self, X: pd.DataFrame):
        result = {}
        
        for variable in self.categories:
            var_values = X[variable].values
            
            for agg_func in self.agg_func:
                col_name = variable if len(self.agg_func) == 1 else f"{variable}_{agg_func}"
                
                # mapの代わりにnumpy配列で処理
                encoding = self.encodings[variable][agg_func]
                prior = self.prior[agg_func]
                
                encoded = np.array([encoding.get(v, prior) for v in var_values], dtype=np.float32)
                
                if self.noise_level > 0:
                    if self.random_state is not None:
                        np.random.seed(self.random_state)
                    encoded = self.add_noise(encoded, self.noise_level)
                
                result[col_name] = encoded
        
        # 結果を一度にDataFrameに変換
        if len(self.agg_func) == 1:
            Xt = X.copy()
            for col_name, values in result.items():
                Xt[col_name] = values
        else:
            non_cat_cols = [col for col in X.columns if col not in self.categories]
            Xt = X[non_cat_cols].copy() if non_cat_cols else pd.DataFrame(index=X.index)
            for col_name, values in result.items():
                Xt[col_name] = values
        
        return Xt
    
    def fit_transform(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for TargetEncode.fit_transform()")
        
        if isinstance(self.categories, str) and self.categories == 'auto':
            self.categories = X.select_dtypes(include=['object', 'category']).columns.tolist()
        elif isinstance(self.categories, str):
            self.categories = [self.categories]
        
        if isinstance(self.cv, int):
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            splits = cv_splitter.split(X, y)
        else:
            if self.groups is None:
                raise ValueError("groups must be provided when cv is not int")
            splits = self.cv.split(X, y, groups=self.groups)
        
        # 結果を格納する辞書（numpy配列で保持）
        encoded_arrays = {}
        y_values = y.values if hasattr(y, 'values') else y
        
        for train_idx, val_idx in splits:
            X_train_values = {col: X[col].values[train_idx] for col in self.categories}
            y_train = y_values[train_idx]
            X_val_values = {col: X[col].values[val_idx] for col in self.categories}
            
            enc_dict, prior_dict = self._fit_fold(X_train_values, y_train)
            
            for variable in self.categories:
                for agg_func in self.agg_func:
                    col_name = variable if len(self.agg_func) == 1 else f"{variable}_{agg_func}"
                    
                    if col_name not in encoded_arrays:
                        encoded_arrays[col_name] = np.empty(len(X), dtype=np.float32)
                    
                    encoding = enc_dict[variable][agg_func]
                    prior = prior_dict[agg_func]
                    var_vals = X_val_values[variable]
                    
                    encoded = np.array([encoding.get(v, prior) for v in var_vals], dtype=np.float32)
                    
                    if self.noise_level > 0:
                        if self.random_state is not None:
                            np.random.seed(self.random_state)
                        encoded = self.add_noise(encoded, self.noise_level)
                    
                    encoded_arrays[col_name][val_idx] = encoded
        
        # DataFrameに変換
        if len(self.agg_func) == 1:
            X_encoded = X.copy()
            for col_name, values in encoded_arrays.items():
                X_encoded[col_name] = values
        else:
            non_cat_cols = [col for col in X.columns if col not in self.categories]
            X_encoded = X[non_cat_cols].copy() if non_cat_cols else pd.DataFrame(index=X.index)
            for col_name, values in encoded_arrays.items():
                X_encoded[col_name] = values
        
        self.fit(X, y)
        return X_encoded
    
    def _fit_fold(self, X_dict, y):
        enc_dict = {}
        prior_dict = {}

        for variable, var_values in X_dict.items():
            enc_dict[variable] = {}
            
            for agg_func in self.agg_func:
                prior = getattr(np, agg_func)(y)
                
                df_agg = pd.DataFrame({'variable': var_values, 'target': y})
                stats = df_agg.groupby('variable', observed=True, sort=False)['target'].agg([agg_func, 'count', 'var'])
                if agg_func == 'std':
                        stats = stats.fillna(0)
                
                m = self.smooth
                if self.smooth == 'auto':
                    # Empirical Bayes smoothing
                    stats['var'] = stats['var'].fillna(0)
                    variance_between = stats[agg_func].var()
                    avg_variance_within = stats['var'].mean()
                    if variance_between > 0:
                        m = avg_variance_within / variance_between
                    else:
                        m = 0  # No smoothing if no variance between groups
                encodings = (stats['count'].values * stats[agg_func].values + m * prior) / (stats['count'].values + m)
                
                #手動パラメータによる平滑化
                # smoothing = 1 / (1 + np.exp(-(stats['count'].values - self.k) / self.f))
                # encodings = prior * (1 - smoothing) + stats[agg_func].values * smoothing
                
                enc_dict[variable][agg_func] = dict(zip(stats.index, encodings))
                prior_dict[agg_func] = prior
        
        return enc_dict, prior_dict