import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter

data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']

print(f"Before SMOTE: {Counter(y_train)}")

smote=SMOTE(random_state=1)

X_train_smote,y_train_smote=smote.fit_resample(X_train_scaled,y_train)

print(f"After SMOTE: {Counter(y_train_smote)}")

resampled_data = {
    'X_train_resampled': X_train_smote,
    'X_test_scaled': data['X_test_scaled'],
    'y_train_resampled': y_train_smote, 
    'y_test': data['y_test']
}

joblib.dump(resampled_data, r'data\processed\smote_data.pkl')
print("Scaling complete. Final data saved to folder!")

