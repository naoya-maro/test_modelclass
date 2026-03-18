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



class MyDataset(Dataset):
    def __init__(self, X, y=None, group_ids=None, is_test=False, window_length=5, polyorder=3):
        self.X = X
        self.y = y
        self.group_ids = group_ids
        self.is_test = is_test
        self.window_length = window_length
        self.polyorder = polyorder

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        spectrum = self.X[idx]
        snv_feat = self._apply_snv(spectrum)
        savgol_feat = savgol_filter(snv_feat, window_length=self.window_length, polyorder=self.polyorder, deriv=2)
        savgol_feat = (savgol_feat - np.mean(savgol_feat)) / np.std(savgol_feat)
        X_tensor = torch.tensor(savgol_feat, dtype=torch.float).unsqueeze(0)
        
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

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv15 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, padding=7 ),
            nn.LayerNorm(normalized_shape=[32, 1555]),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LayerNorm(normalized_shape=[64, 1555]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LayerNorm(normalized_shape=[128, 777]),
            nn.GELU(),
            # nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(128),
            nn.Conv1d(128, 16, kernel_size=1),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
                nn.Linear(16 * 128, 256), #* (input // 4)
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64), #* (input // 4)
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
        )
    
    def forward(self, x):
        x = self.conv15(x)
        return self.fc(x)




class CNN1DModel(ModelInterface):
    '''1dCNNのモデルクラス'''
    def __init__(self, params: dict):
        super().__init__(params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = self.params.get('epochs', 50)
        self.lr = self.params.get('lr', 0.001)
        self.batch_size = self.params.get('batch_size', 32)
        self.random_state = self.params.get('random_state', 42)
        self.early_stopping_rounds = self.params.get('early_stopping_rounds', 10)
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self._set_seed()
        self.group_ids = kwargs.get('group_ids')
        
        self.model = CNN1D().to(self.device)
        self.optimizer = RAdamScheduleFree(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999)) #optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.HuberLoss(delta=0.5) #nn.MSELoss()
        train_loader = self._create_dataloader(X_train, y=y_train, group_ids=self.group_ids)
        valid_loader = self._create_dataloader(X_val, y=y_val, group_ids=self.group_ids) if X_val is not None else None
        
        best_loss = float('inf')
        early_stopping_counter = 0
        self.history = np.zeros((0, 3)) # epoch, train_loss, val_loss
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.train()
            train_loss = 0
            for X_batch, y_batch, species_number in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                if np.random.random() > 0.0:
                    X_batch, y_a, y_b, lam = self.mixup_data(X_batch, y_batch, species_number)
                    outputs = self.model(X_batch)
                    loss = self.mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            if valid_loader is not None:
                self.model.eval()
                self.optimizer.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch, species_number in valid_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                avg_val_loss = val_loss / len(valid_loader.dataset)
                        
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stopping_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= self.early_stopping_rounds:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            print(f'Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
            self.history = np.vstack([self.history, [epoch+1, avg_train_loss, avg_val_loss]])
        return self
    
    def predict(self, X) -> np.ndarray:
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        self.optimizer.eval()
        test_loader = self._create_dataloader(X, is_test=True)
        predictions = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
        return np.vstack(predictions).flatten()
    
    def _create_dataloader(self, X, y=None, group_ids=None, is_test=False):
        dataset = MyDataset(X, y=y, group_ids=group_ids, is_test=is_test)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=not is_test, num_workers=4, pin_memory=True)
    
    def _set_seed(self) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms = True
    
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
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        # 損失関数も混ぜ合わせる割合に応じて計算
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
            