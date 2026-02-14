import pandas as pd
import numpy as np

def apply_feature_engineering(df):
    df=df.copy()
    # 1. Interaction between top 2 features (V14 & V4)
    df['V14_V4_interaction'] = df['V14'] * df['V4']
    
    # # 3. Squaring V14 to penalize extreme deviations (common in fraud)
    # df['V14_squared'] = df['V14'] ** 2
    
    # 4. Drop the low-importance 'noise' features we identified
    # This helps the model focus on the signals that matter
    noise_features = ['V24', 'V26', 'V27','V22','V23','V15','V25'] 
    df = df.drop(columns=[col for col in noise_features if col in df.columns])
    
    return df
