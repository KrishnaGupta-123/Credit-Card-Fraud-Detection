from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import plot_confusion_matrix
import pandas as pd

# Before Smote
data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']
data1=joblib.load('data/processed/fraud_data_split.pkl')
X_train=data1['X_train']


ranf = RandomForestClassifier(
    n_estimators=100,      
    max_depth=20,          
    random_state=42,
    n_jobs=-1,             
    verbose=1
)

ranf.fit(X_train_scaled, y_train)
y_pred_scaled = ranf.predict(X_test_scaled)

plot_confusion_matrix(y_test, y_pred_scaled, title='Phase 4: Random Forest + Without_SMOTE')

#After smote
data=joblib.load('data/processed/smote_data.pkl')
X_train_resampled=data['X_train_resampled']
y_train_resampled=data['y_train_resampled']


ranf_smote = RandomForestClassifier(
    n_estimators=100,      
    max_depth=20,          
    random_state=42,
    n_jobs=-1,             
    verbose=1
)

ranf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_smote = ranf_smote.predict(X_test_scaled)

plot_confusion_matrix(y_test, y_pred_smote, title='Phase 4: Random Forest + SMOTE')

importances_scaled = ranf.feature_importances_
importances_smote = ranf_smote.feature_importances_

feature_names = X_train.columns
rf_feat_imp = pd.DataFrame({'feature': feature_names, 
                            'Importances_scaled': importances_scaled,
                            'Importances_smote': importances_smote})
rf_feat_imp = rf_feat_imp.sort_values(by='Importances_scaled', ascending=False)

print(rf_feat_imp)

# Save the trained model
model_path = 'models/RanFor_scaled_21_5.pkl'
model_path_bst='models/RanFor_smote_18_11.pkl'
joblib.dump(ranf, model_path)
joblib.dump(ranf_smote,model_path_bst )

print(f"✅ Model successfully exported !!!")