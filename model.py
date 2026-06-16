from abc import ABC, abstractmethod
import numpy as np
import lightgbm as lgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from interpret.glassbox import ExplainableBoostingRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import tqdm
import torch.optim.optimizer
from schedulefree import RAdamScheduleFree
from scipy.signal import savgol_filter


class ModelInterface(ABC):
    '''モデルのインターフェース'''
    
    def __init__(self, params:dict):
        self.params = params
        self.model = None
        
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X_train):
        pass



class LightGBMModel(ModelInterface):
    '''LightGBMのモデルクラス'''
         
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        weight_train = kwargs.get('weight_train')
        weight_val = kwargs.get('weight_val')
        stopping_rounds = kwargs.get('stopping_rounds', 100)
        log_evaluation = kwargs.get('log_evaluation', 100)
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=weight_train)
        valid_data = lgb.Dataset(X_val, label=y_val, weight=weight_val) if X_val is not None else None
        
        callbacks = [
            lgb.early_stopping(stopping_rounds),
            lgb.log_evaluation(log_evaluation),
            # lgb.reset_parameter(learning_rate=learning_rate_scheduler)
            ]
        
        valid_sets = [train_data]
        valid_names = ['train']
        if valid_data:
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        self.model = lgb.train(self.params, train_data, valid_sets=valid_sets,
                               valid_names=valid_names,
                               callbacks=callbacks
                               )
    
    def predict(self, X):
        return self.model.predict(X)
    
class PLSModel(ModelInterface):
    '''PLSのモデルクラス'''
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model = PLSRegression(**self.params)
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X).flatten()

class ExtraTreesModel(ModelInterface):
    '''ExtraTreesのモデルクラス'''
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model = ExtraTreesRegressor(**self.params)
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)

class EBMModel(ModelInterface):
    '''EBMのモデルクラス'''
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model = ExplainableBoostingRegressor(**self.params)
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)



class SpectrumAugmenter:
    def __init__(self, p=0.5, seed=42):
        self.p = p
        self.rng = np.random.RandomState(seed)

    def add_noise(self, x, std=0.001):
        """ガウシアンノイズを付与"""
        noise = self.rng.normal(0, std, size=x.shape)
        return x + noise

    def random_rescale(self, x, low=0.995, high=1.005):
        """全体の振幅をスケーリング（光源の強さの変動）"""
        scale = self.rng.uniform(low, high)
        return x * scale

    def random_shift(self, x, max_shift=3):
        """波長方向にわずかにずらす（温度変化や位置ズレのシミュレート）"""
        shift = self.rng.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return x
        
        if shift > 0:
            return np.pad(x, (shift, 0), mode='edge')[:-shift]
        else:
            abs_shift = abs(shift)
            return np.pad(x, (0, abs_shift), mode='edge')[abs_shift:]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ランダムに拡張を適用"""
        if self.rng.rand() > self.p:
            return x
        x = self.add_noise(x)
        x = self.random_rescale(x)
        x = self.random_shift(x)
        return x

class MyDataset(Dataset):
    def __init__(self, X, y=None, group_ids=None, ch_size=2, is_test=False, window_length=15, polyorder=3, transforms=None):
        self.X = X
        self.y = y
        self.group_ids = group_ids
        self.ch_size = ch_size
        self.is_test = is_test
        self.window_length = window_length
        self.polyorder = polyorder
        self.transforms = transforms

    def __len__(self): 
        return len(self.X)
    
    def __getitem__(self, idx):
        spectrum = self.X[idx]
        if self.transforms is not None:
            spectrum = self.transforms(spectrum)
            
        if self.ch_size == 2:
            spectrum_smooth = savgol_filter(spectrum, window_length=self.window_length, polyorder=self.polyorder, deriv=2)
            spectrum_smooth = self._apply_snv(spectrum_smooth)
            spectrum = np.stack([spectrum, spectrum_smooth], axis=0)
            X_tensor = torch.tensor(spectrum, dtype=torch.float)
        elif self.ch_size == 1:
            X_tensor = torch.tensor(spectrum, dtype=torch.float).unsqueeze(0)
        elif self.ch_size == 0:
            X_tensor = torch.tensor(spectrum, dtype=torch.float)
        
        if self.is_test:
            return X_tensor
        else:
            y = self.y[idx]
            y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(0)
            species_number = self.group_ids[idx]
            species_number = torch.tensor(species_number, dtype=torch.float)
            return X_tensor, y_tensor, species_number
    
    def _apply_snv(self, x):
        return (x - np.mean(x)) / np.std(x)

class EnsembleLossWrapper(nn.Module):
    """
    モデルの出力がアンサンブル（3次元）か通常（2次元）かを自動判別し、
    次元の平坦化と正解ラベルの拡張を裏側で隠蔽するラッパーLoss
    """
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, out0, y, out1, species_number, epoch):
        # 💡 out0 が3次元 (B, E, D) ＝ アンサンブルモデルの場合だけ自動変形
        if len(out0.shape) == 3:
            B, E, D1 = out0.shape
            _, _, D2 = out1.shape
            
            # 裏側でひっそり次元を合わせて、元のLoss関数に丸投げする
            out0_flat = out0.reshape(B * E, D1)
            out1_flat = out1.reshape(B * E, D2)
            y_flat = y.unsqueeze(1).expand(-1, E, -1).reshape(B * E, -1)
            species_flat = species_number.unsqueeze(1).expand(-1, E).reshape(B * E)
            
            return self.base_criterion(out0_flat, y_flat, out1_flat, species_flat, epoch)
        
        # 通常モデル（2次元）の場合は、何もせずそのまま元のLoss関数を実行
        return self.base_criterion(out0, y, out1, species_number, epoch)
    
class EnsembleLossWrapperV2(nn.Module):
    """
    モデルの出力がアンサンブル（3次元）か通常（2次元）かを自動判別し、
    次元の平坦化と正解ラベルの拡張を裏側で隠蔽するラッパーLoss
    """
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, out0, y):
        # 💡 out0 が3次元 (B, E, D) ＝ アンサンブルモデルの場合だけ自動変形
        if len(out0.shape) == 3:
            B, E, D1 = out0.shape
            
            # 裏側でひっそり次元を合わせて、元のLoss関数に丸投げする
            if D1 == 1:
                out0_flat = out0.reshape(B * E, D1)
                y_flat = y.unsqueeze(1).expand(-1, E, -1).reshape(B * E, -1)
            else:
                out0_flat = out0.reshape(B * E, D1)
                y_flat = y.unsqueeze(1).expand(-1, E).reshape(B * E)
            
            return self.base_criterion(out0_flat, y_flat)
        
        # 通常モデル（2次元）の場合は、何もせずそのまま元のLoss関数を実行
        return self.base_criterion(out0, y)


class ScheduleMultiTaskLoss(nn.Module):
    def __init__(self, start_alpha=0.5, end_alpha=0.05, max_epochs=50):
        super().__init__()
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.max_epochs = max_epochs
        
        self.reg_criterion = nn.HuberLoss(delta=1.0) # MSEより頑健
        self.cls_criterion = nn.CrossEntropyLoss()
        self.loss_logs = {"reg": 0.0, "cls": 0.0, "alpha": 0.0}

    def forward(self, pred_reg, target_reg, pred_cls, target_cls, current_epoch):
        # エポックに応じて alpha を徐々に小さくする（線形減衰）
        if self.start_alpha == 0:
            self.alpha = self.start_alpha
        else:
            alpha = self.start_alpha - (self.start_alpha - self.end_alpha) * (current_epoch / self.max_epochs)
            self.alpha = max(alpha, self.end_alpha)
        
        loss_reg = self.reg_criterion(pred_reg, target_reg)
        loss_cls = self.cls_criterion(pred_cls, target_cls)
        
        self.current_reg_loss = loss_reg.item()
        self.current_cls_loss = loss_cls.item()
        
        return loss_reg + self.alpha * loss_cls

class CNN1DModel(ModelInterface):
    '''1dCNNのモデルクラス'''
    def __init__(self, params: dict):
        super().__init__(params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        cfg = params.get('Training', {})
        self.ch_size = cfg.get('ch_size', 2)
        self.epochs = cfg.get('epochs', 50)
        self.lr = cfg.get('lr', 0.001)
        self.batch_size = cfg.get('batch_size', 32)
        self.random_state = cfg.get('random_state', 42)
        self.early_stopping_rounds = cfg.get('early_stopping_rounds', 10)
        self.weight_decay = cfg['weight_decay']
        self.loss_alpha = cfg['loss_alpha']
        self.augmenter_p = cfg['augmenter_p']
        self.mixup_p = cfg['mixup_p']
        
        self.best_model_state = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self._set_seed()
        self.group_ids = kwargs.get('group_ids')
        # self.group_ids_unique = len(np.unique(self.group_ids))
        self.params['n_class'] = int(len(np.unique(self.group_ids)))
        
        self.model = CNN.ScaleAdaptiveMambaV2(self.params).to(self.device)
        self.optimizer = RAdamScheduleFree(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        # self.criterion = EnsembleLossWrapper(ScheduleMultiTaskLoss(start_alpha=self.loss_alpha))
        train_loader = self._create_dataloader(X_train, y=y_train, group_ids=self.group_ids, transforms=SpectrumAugmenter(p=self.augmenter_p))
        valid_loader = self._create_dataloader(X_val, y=y_val, group_ids=self.group_ids, transforms=None) if X_val is not None else None
        
        best_loss = float('inf')
        early_stopping_counter = 0
        self.history = np.zeros((0, 3)) # epoch, train_loss, val_loss
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.train()
            train_loss = 0
            for X_batch, y_batch, species_number in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False):
                X_batch, y_batch, species_number = X_batch.to(self.device), y_batch.to(self.device), species_number.to(self.device)
                self.optimizer.zero_grad()
                
                if np.random.rand() > self.mixup_p:
                    X_batch, y_batch = self.mixup_data(X_batch, y_batch, species_number)
                    outputs = self.model(X_batch)
                    # loss = self.criterion(outputs[0], y_batch, outputs[1], species_number.long(), epoch)
                    # 2. それぞれの生の損失を計算
                    loss = self.model.compute_multitask_loss(outputs[0], y_batch, outputs[1], species_number.long())
                else:
                    outputs = self.model(X_batch)
                    # loss = self.criterion(outputs[0], y_batch, outputs[1], species_number.long(), epoch)
                    loss = self.model.compute_multitask_loss(outputs[0], y_batch, outputs[1], species_number.long())
                loss.backward()                
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            if valid_loader is not None:
                self.model.eval()
                self.optimizer.eval()
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for X_batch, y_batch, species_number in valid_loader:
                        X_batch, y_batch, species_number = X_batch.to(self.device), y_batch.to(self.device), species_number.to(self.device)
                        
                        outputs = self.model(X_batch)
                        pred_log = outputs[0].mean(dim=1)
                        val_preds.append(pred_log.detach().cpu().numpy())
                        val_targets.append(y_batch.detach().cpu().numpy())
                val_preds = np.vstack(val_preds).flatten()
                val_targets = np.vstack(val_targets).flatten()
                val_preds = np.clip(np.expm1(val_preds), 0.5, None)
                val_targets = np.expm1(val_targets)
                avg_val_loss = np.sqrt(np.mean((val_preds - val_targets) ** 2))
                        
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stopping_counter = 0
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= self.early_stopping_rounds:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            print(f'Train Loss: {avg_train_loss:.4f} - Val RMSE: {avg_val_loss:.4f}')
            self.history = np.vstack([self.history, [epoch+1, avg_train_loss, avg_val_loss]])
        
        # メモリを解放
        self.optimizer.eval()
        self.model.load_state_dict(self.best_model_state)
        self.model.to('cpu')
        
        del self.optimizer, train_loader, valid_loader#, self.criterion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self
    
    def predict(self, X) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()
        test_loader = self._create_dataloader(X, is_test=True)
        predictions = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(self.device)
                
                outputs = self.model(X_batch)
                if len(outputs[0].shape) == 3:
                    pred = outputs[0].mean(dim=1).cpu().numpy()
                else:
                    pred = outputs[0].cpu().numpy()
                predictions.append(pred)
        
        self.model.to('cpu')
        del test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return np.vstack(predictions).flatten()
    
    def _create_dataloader(self, X, y=None, group_ids=None, is_test=False, transforms=None, batch_size=None):
        dataset = MyDataset(X, y=y, group_ids=group_ids, ch_size=self.ch_size, is_test=is_test, transforms=transforms)
        batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(dataset, batch_size=batch_size, shuffle=not is_test, num_workers=4, pin_memory=True)
    
    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # torch Generator（mixup の分岐に使用）
        self._torch_gen = torch.Generator()
        self._torch_gen.manual_seed(self.random_state)

        # numpy Generator（mixup の beta サンプリングに使用）
        self._np_rng = np.random.default_rng(self.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
    
    def save_model(self, path):
        torch.save(self.best_model_state, path)
    
    def mixup_data(self, x, y, group_ids, alpha=0.4):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            # ベータ分布から混ぜ合わせる割合(lam)を決定
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.arange(batch_size).to(x.device)
        unique_groups = torch.unique(group_ids)
        
        for k in unique_groups:
            # このグループに属するサンプルのマスク
            mask = (group_ids == k)
            idx_in_group = index[mask]
            if len(idx_in_group) > 1:
            # グループ内だけでシャッフル
                perm = torch.randperm(len(idx_in_group))
                index[mask] = idx_in_group[perm]

        # 混ぜ合わせる
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        y_linear_a = torch.exp(y_a)
        y_linear_b = torch.exp(y_b)
        y_mixed_linear = lam * y_linear_a + (1 - lam) * y_linear_b
        y_mixed_log = torch.log(y_mixed_linear)
        return mixed_x, y_mixed_log


class RealMLPModel(ModelInterface):
    '''RealMLPのモデルクラス'''
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model = RealMLP_TD_Regressor(**self.params)
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
            
