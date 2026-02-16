import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from sklearn.utils.class_weight import compute_class_weight
from utils import plot_confusion_matrix
import numpy as np

data=joblib.load('data/processed/final_scaled_data.pkl')
X_train_scaled=data['X_train_scaled']
y_train=data['y_train']
X_test_scaled=data['X_test_scaled']
y_test=data['y_test']

base_ae = load_model(r'models\best_autoencoder_12fn.h5', compile=False)

encoder_output = base_ae.layers[2].output
encoder_model = Model(inputs=base_ae.input, outputs=encoder_output)
encoder_model.trainable = False

x = Dense(16, activation='relu',name="classifier_hidden")(encoder_model.output)
prediction = Dense(2, activation='softmax', name="classifier_output")(x)
classifier = Model(inputs=encoder_model.input, outputs=prediction)

f1_metric = tf.keras.metrics.F1Score(average='weighted', name='f1_score')
classifier.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy',f1_metric])

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
# custom_weights = {0: 1.0, 1: 600.0}
classifier.fit(X_train_scaled, y_train, 
               epochs=100, 
               batch_size=256, 
               class_weight=class_weights,
               validation_split=0.1,
               verbose=1)

y_pred_probs = classifier.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)
plot_confusion_matrix(y_test,y_pred)