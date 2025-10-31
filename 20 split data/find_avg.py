import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL1_PATH = "saved_models/mnist_model_v1.h5"
MODEL2_PATH = "saved_models/mnist_model_v2.h5"
MODEL3_PATH = "saved_models/mnist_model_v3.h5"
MODEL4_PATH = "saved_models/mnist_model_v4.h5"
MODEL5_PATH = "saved_models/mnist_model_v5.h5"
MODEL6_PATH = "saved_models/mnist_model_v6.h5"
MODEL7_PATH = "saved_models/mnist_model_v7.h5"
MODEL8_PATH = "saved_models/mnist_model_v8.h5"
MODEL9_PATH = "saved_models/mnist_model_v9.h5"
MODEL10_PATH = "saved_models/mnist_model_v10.h5"
MODEL11_PATH = "saved_models/mnist_model_v11.h5"
MODEL12_PATH = "saved_models/mnist_model_v12.h5"
MODEL13_PATH = "saved_models/mnist_model_v13.h5"
MODEL14_PATH = "saved_models/mnist_model_v14.h5"
MODEL15_PATH = "saved_models/mnist_model_v15.h5"
MODEL16_PATH = "saved_models/mnist_model_v16.h5"
MODEL17_PATH = "saved_models/mnist_model_v17.h5"
MODEL18_PATH = "saved_models/mnist_model_v18.h5"
MODEL19_PATH = "saved_models/mnist_model_v19.h5"
MODEL20_PATH = "saved_models/mnist_model_v20.h5"

m1 = keras.models.load_model(MODEL1_PATH)
m2 = keras.models.load_model(MODEL2_PATH)
m3 = keras.models.load_model(MODEL3_PATH)
m4 = keras.models.load_model(MODEL4_PATH)
m5 = keras.models.load_model(MODEL5_PATH)
m6 = keras.models.load_model(MODEL6_PATH)
m7 = keras.models.load_model(MODEL7_PATH)
m8 = keras.models.load_model(MODEL8_PATH)
m9 = keras.models.load_model(MODEL9_PATH)
m10 = keras.models.load_model(MODEL10_PATH)
m11 = keras.models.load_model(MODEL11_PATH)
m12 = keras.models.load_model(MODEL12_PATH)
m13 = keras.models.load_model(MODEL13_PATH)
m14 = keras.models.load_model(MODEL14_PATH)
m15 = keras.models.load_model(MODEL15_PATH)
m16 = keras.models.load_model(MODEL16_PATH)
m17 = keras.models.load_model(MODEL17_PATH)
m18 = keras.models.load_model(MODEL18_PATH)
m19 = keras.models.load_model(MODEL19_PATH)
m20 = keras.models.load_model(MODEL20_PATH)

w1 = m1.get_weights()
w2 = m2.get_weights()   
w3 = m3.get_weights()
w4 = m4.get_weights()
w5 = m5.get_weights()
w6 = m6.get_weights()
w7 = m7.get_weights()
w8 = m8.get_weights()
w9 = m9.get_weights()
w10 = m10.get_weights()
w11 = m11.get_weights()
w12 = m12.get_weights()
w13 = m13.get_weights()
w14 = m14.get_weights()
w15 = m15.get_weights()
w16 = m16.get_weights()
w17 = m17.get_weights()
w18 = m18.get_weights()
w19 = m19.get_weights()
w20 = m20.get_weights()

avg_weights = []
for idx, (a, b, c, d, e, f, g, h, i_w, j, k, l, m, n, o, p, q, r, s, t) in enumerate(zip(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20)):
    if not all(x.shape == a.shape for x in [b, c, d, e, f, g, h, i_w, j, k, l, m, n, o, p, q, r, s, t]):
        raise ValueError(f"Weight shape mismatch at layer {idx}")
    avg_weights.append((a + b + c + d + e + f + g + h + i_w + j + k + l + m + n + o + p + q + r + s + t) / 20.0)


avg_model = keras.models.clone_model(m1)

avg_model.set_weights(avg_weights)

avg_model.save_weights("saved_models\\averaged.weights.h5")

print("Saved: averaged_weights.h5")