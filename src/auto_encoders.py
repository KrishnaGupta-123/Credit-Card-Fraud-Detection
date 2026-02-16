import os
import joblib
import pandas as pd
import numpy as np
from utils import plot_training_history,plot_confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
 
data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

X_train_scaled_normal=X_train_scaled[y_train==0]
input_dim = X_train_scaled_normal.shape[1] 

# 1. Encoder (Compression)
input_layer = Input(shape=(input_dim,))
encoder = Dense(10, activation="tanh")(input_layer)
encoder = Dense(5, activation="leaky_relu")(encoder)

# 2. Decoder (Reconstruction)
decoder = Dense(10, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='linear')(decoder)

# 3. Full Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

#Callback for saving the best model
checkpoint = ModelCheckpoint(
    'autoencoder_fn.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min',
    verbose=1
)
history = autoencoder.fit(
    X_train_scaled_normal, X_train_scaled_normal, 
    epochs=150,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test_scaled, X_test_scaled), 
    verbose=1,
    callbacks=[checkpoint]
).history

reconstructions = autoencoder.predict(X_test_scaled)

mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)

# Calculate error on the NORMAL training data to find the "normal" limit
train_reconstructions = autoencoder.predict(X_train_scaled_normal)
train_mse = np.mean(np.power(X_train_scaled_normal - train_reconstructions, 2), axis=1)

threshold = np.percentile(train_mse, 95) 
print(f"Reconstruction Error Threshold: {threshold}")

test_fraud_mse = mse[y_test == 1]
test_normal_mse = mse[y_test == 0]
print(f"Average Normal MSE: {np.mean(test_normal_mse):.4f}")
print(f"Average Fraud MSE: {np.mean(test_fraud_mse):.4f}")

# If error > threshold, it's flagged as Fraud (1)
y_pred_ae = [1 if e > threshold else 0 for e in mse]

plot_confusion_matrix(y_test, y_pred_ae, 
                      title='Phase 5: Autoencoder Anomaly Detection')

model = load_model('best_autoencoder_12fn.h5',compile=False)

train_reconstructions = model.predict(X_train_scaled_normal)
train_mse = np.mean(np.square(X_train_scaled_normal - train_reconstructions), axis=1)
threshold = np.percentile(train_mse, 95)

reconstructions = model.predict(X_test_scaled)
test_mse = np.mean(np.square(X_test_scaled - reconstructions), axis=1)

y_pred = [1 if e > threshold else 0 for e in test_mse]

print(plot_confusion_matrix(y_test, y_pred))
