import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL1_PATH = "D:\\amrita intern\\5 split data\\saved_models\\weight_exchg_1_v1.h5"
MODEL2_PATH = "D:\\amrita intern\\5 split data\\saved_models\\weight_exchg_1_v2.h5"
MODEL3_PATH = "D:\\amrita intern\\5 split data\\saved_models\\weight_exchg_1_v3.h5"
MODEL4_PATH = "D:\\amrita intern\\5 split data\\saved_models\\weight_exchg_1_v4.h5"
MODEL5_PATH = "D:\\amrita intern\\5 split data\\saved_models\\weight_exchg_1_v5.h5"


m1 = keras.models.load_model(MODEL1_PATH)
m2 = keras.models.load_model(MODEL2_PATH)
m3 = keras.models.load_model(MODEL3_PATH)
m4 = keras.models.load_model(MODEL4_PATH)
m5 = keras.models.load_model(MODEL5_PATH)

w1 = m1.get_weights()
w2 = m2.get_weights()   
w3 = m3.get_weights()
w4 = m4.get_weights()
w5 = m5.get_weights()  

avg_weights = []
for i, (a, b, c, d, e) in enumerate(zip(w1, w2, w3, w4, w5)):
    if a.shape != b.shape:
        raise ValueError(f"Weight shape mismatch at index {i}: {a.shape} vs {b.shape}")
    avg_weights.append((a + b + c + d + e) / 5.0)


avg_model = keras.models.clone_model(m1)

avg_model.set_weights(avg_weights)

avg_model.save_weights("D:\\amrita intern\\5 split data\\saved_models\\averaged.weights.h5")

print("Saved: averaged_weights.h5")