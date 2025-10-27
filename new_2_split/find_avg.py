import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL1_PATH = "saved_models\mnist_model_v1.h5"
MODEL2_PATH = "saved_models\mnist_model_v2.h5"


m1 = keras.models.load_model(MODEL1_PATH)
m2 = keras.models.load_model(MODEL2_PATH)


w1 = m1.get_weights()
w2 = m2.get_weights()   


avg_weights = []
for idx, (a, b) in enumerate(zip(w1, w2)):
    if not all(x.shape == a.shape for x in [b]):
        raise ValueError(f"Weight shape mismatch at layer {idx}")
    avg_weights.append((a + b) / 2.0)


avg_model = keras.models.clone_model(m1)

avg_model.set_weights(avg_weights)

avg_model.save_weights("saved_models\\averaged.weights.h5")

print("Saved: averaged_weights.h5")