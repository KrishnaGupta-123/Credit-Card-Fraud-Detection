from sklearn.ensemble import RandomForestClassifier
import joblib
from confusion_matrix import plot_confusion_matrix

# Before Smote
data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']


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
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

ranf = RandomForestClassifier(
    n_estimators=100,      
    max_depth=20,          
    random_state=42,
    n_jobs=-1,             
    verbose=1
)

ranf.fit(X_train_resampled, y_train_resampled)
y_pred_smote = ranf.predict(X_test_scaled)

plot_confusion_matrix(y_test, y_pred_smote, title='Phase 4: Random Forest + SMOTE')
