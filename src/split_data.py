# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# This moves one level up to the root and then into data/raw/
data_path = os.path.join(BASE_DIR, '..', 'data', 'raw', 'creditcard.csv')

df = pd.read_csv(data_path)
# %%
# df=pd.read_csv(r"data\raw\creditcard.csv")

# %%
# imbalanced Dataset 
df['Class'].value_counts()
# %%
df.info()
# %%
X=df.drop(['Class'],axis=1)
y=df['Class']

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# %%
os.makedirs('data/processed', exist_ok=True)
data_bundle = {
    'X_train': X_train, 'X_test': X_test, 
    'y_train': y_train, 'y_test': y_test
}
joblib.dump(data_bundle, 'data/processed/fraud_data_split.pkl')
print("Data prepared and saved to data/processed/!")
# %%
