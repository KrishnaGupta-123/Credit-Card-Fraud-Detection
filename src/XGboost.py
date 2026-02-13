
import xgboost as xgb
from xgboost import plot_importance
from confusion_matrix import plot_confusion_matrix
import joblib
import matplotlib.pyplot as plt

# 1.Load data
data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

# 2. Initialize XGBoost with GPU acceleration
# We use scale_pos_weight to handle the 0.17% fraud ratio
ratio = float(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist', #making bins so instead of looking for indivisual data points 
                        # it looks at sets of them.It then only tests the boundaries
                        #  between these "sets" to find the best split.
    device='cuda',      # Targeting the GPU
    scale_pos_weight=ratio, # To penalize the algo for flagging fraud as normal
    random_state=42
)

# 3. Train
xgb_model.fit(X_train_scaled, y_train)

# 4. Predict and Evaluate
y_pred = xgb_model.predict(X_test_scaled)
plot_confusion_matrix(y_test, y_pred, title='Phase 4: XGBoost (GPU Accelerated)')

# 5.Feature Importance
plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.show()