import numpy as np
from tensorflow import keras

def evaluate_model(X_test, y_test, model_load_path):
    model = keras.models.load_model(model_load_path)
    y_pred = model.predict(X_test).flatten()
    mae = np.mean(np.abs(y_test - y_pred))
    return mae
