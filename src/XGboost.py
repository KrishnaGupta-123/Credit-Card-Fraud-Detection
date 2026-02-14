
import xgboost as xgb
from xgboost import plot_importance
from utils import plot_confusion_matrix
import joblib
import matplotlib.pyplot as plt
import os

# 1.Load data
data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

# 2. Initialize XGBoost with GPU acceleration
# We use scale_pos_weight to handle the 0.17% fraud ratio
ratio = float(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
best_params={'n_estimators': 920, 
             'max_depth': 3, 
             'learning_rate': 0.03914927724898531, 
             'subsample': 0.832548161367491, 
             'colsample_bytree': 0.6482287084307642, 
             'reg_alpha': 0.0015411953828949893, 
             'reg_lambda': 0.20955843920976702, 
             'scale_pos_weight': 390.99319837409286, 
             'max_delta_step': 8, 
             'tree_method': 'hist', 
             'device': 'cuda', 
             'random_state': 42}

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist', #making bins so instead of looking for indivisual data points 
                        # it looks at sets of them.It then only tests the boundaries
                        #  between these "sets" to find the best split.
    device='cuda',      # Targeting the GPU
    scale_pos_weight=ratio, # To penalize the algo for flagging fraud as normal
    early_stopping_rounds=10,
    eval_metric=["logloss", "aucpr"], # Watch both LogLoss and PR Curve
    random_state=42
)
xgb_model_bst = xgb.XGBClassifier(**best_params)
# 3. Train
xgb_model.fit(X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)], # Data used for monitoring
            verbose=True)  # Prints the progress for each tree

xgb_model_bst.fit(X_train_scaled, y_train)

print(f"Stopped at iteration: {xgb_model.best_iteration}")

# 4. Predict and Evaluate
y_pred = xgb_model.predict(X_test_scaled)
y_pred1 = xgb_model_bst.predict(X_test_scaled)

plot_confusion_matrix(y_test, y_pred, title='Phase 4: XGBoost (GPU Accelerated)')
plot_confusion_matrix(y_test, y_pred1, title='Phase 4: XGBoost (GPU Accelerated) best_params')
# 5.Feature Importance
plot_importance(xgb_model, importance_type='gain')
plot_importance(xgb_model_bst, importance_type='gain')

plt.show()

# Save the trained model
model_path_bst = 'models/champion_xgboost_16_33.pkl'
model_path='models/xgboost_16_53.pkl'
joblib.dump(xgb_model_bst, model_path_bst)
joblib.dump(xgb_model, model_path)

print(f"✅ Model successfully exported !!!")