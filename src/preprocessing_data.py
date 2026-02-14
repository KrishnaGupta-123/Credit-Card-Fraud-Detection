import joblib
import os
from sklearn.preprocessing import StandardScaler

data=joblib.load(r'data\processed\fraud_data_split.pkl')

X_train=data['X_train']
X_test=data['X_test']

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

processed_data = {
    'X_train_scaled': X_train,
    'X_test_scaled': X_test,
    'y_train': data['y_train'], 
    'y_test': data['y_test']
}

joblib.dump(processed_data, r'data\processed\final_scaled_data.pkl')
print("Scaling complete. Final data saved to folder!")