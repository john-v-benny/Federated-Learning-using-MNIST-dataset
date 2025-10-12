# Average weights from two models and save the averaged model/weights

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL1_PATH = "saved_models/mnist_model.h5"       # first model path
MODEL2_PATH = "saved_models/mnist_model2.h5"     # second model path (change if different)

if not os.path.exists(MODEL1_PATH):
    raise FileNotFoundError(f"Missing: {MODEL1_PATH}")
if not os.path.exists(MODEL2_PATH):
    raise FileNotFoundError(f"Missing: {MODEL2_PATH}")

# Load models
m1 = keras.models.load_model(MODEL1_PATH)
m2 = keras.models.load_model(MODEL2_PATH)

# Collect weights
w1 = m1.get_weights()
w2 = m2.get_weights()

# Sanity checks
if len(w1) != len(w2):
    raise ValueError("Models have different number of weight arrays (different architectures).")

avg_weights = []
for i, (a, b) in enumerate(zip(w1, w2)):
    if a.shape != b.shape:
        raise ValueError(f"Weight shape mismatch at index {i}: {a.shape} vs {b.shape}")
    avg_weights.append((a + b) / 2.0)

# Create averaged model with same architecture as model 1
avg_model = keras.models.clone_model(m1)
# Build before setting weights (important for saving)
try:
    avg_model.build(m1.input_shape)
except Exception:
    # Fallback: run a dummy forward pass to build
    dummy = tf.zeros((1,) + tuple(dim if dim is not None else 1 for dim in m1.input_shape[1:]))
    _ = avg_model(dummy)

avg_model.set_weights(avg_weights)

# Save outputs
avg_model.save_weights("saved_models/averaged.weights.h5") # weights only

print("Saved: averaged_weights.h5")