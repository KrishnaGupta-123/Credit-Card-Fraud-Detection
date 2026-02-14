import optuna
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

def objective(trial):
    # 1. Define the Search Space for Bayesian Optimization
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 100, 1000),
        'max_delta_step': trial.suggest_int('max_delta_step', 1, 10),
        'tree_method': 'hist',  
        'random_state': 42,
        'device': 'cuda'
        
    }

    # 2. Train the model with suggested params
    model = XGBClassifier(**params)
    model.fit(X_train_scaled, y_train)
    
    # 3. Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # 4. Define the Goal (Penalty Function)
    score = (fn * 30) + fp 
    return score

# 5. Create the study and run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # 50 trials is usually enough

best_params=study.best_params
best_params.update({'tree_method': 'hist', 'device': 'cuda', 'random_state': 42})
print("Best Hyperparameters:",best_params)
print("Best Score (Penalty):", study.best_value)