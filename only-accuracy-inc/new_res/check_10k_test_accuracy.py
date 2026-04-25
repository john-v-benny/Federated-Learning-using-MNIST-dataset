import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


def preprocess_inputs(x, input_shape):
    x = x.astype("float32")
    # If pixel range looks like [0, 255], normalize to [0, 1].
    if x.max() > 1.5:
        x = x / 255.0

    # Adapt to the model's expected input format.
    # Common cases: (None, 784), (None, 28, 28), (None, 28, 28, 1)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 2:
        x = x.reshape(x.shape[0], -1)
    elif len(input_shape) == 3:
        # Model expects (N, H, W), keep as is.
        pass
    elif len(input_shape) == 4:
        # Model expects channels, ensure channel axis exists.
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
    else:
        raise ValueError(f"Unsupported model input shape: {input_shape}")

    return x


def evaluate_model(model_path, test_npz_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(test_npz_path):
        raise FileNotFoundError(f"Test dataset file not found: {test_npz_path}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"Loading test dataset from: {test_npz_path}")
    test_data = np.load(test_npz_path)

    if "x" not in test_data or "y" not in test_data:
        raise KeyError("NPZ file must contain 'x' and 'y' arrays.")

    x_test = preprocess_inputs(test_data["x"], model.input_shape)
    y_test = test_data["y"]

    print(f"x_test shape after preprocessing: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Try the model's own compiled metrics first.
    try:
        results = model.evaluate(x_test, y_test, verbose=0)
        if isinstance(results, (list, tuple)):
            metrics = dict(zip(model.metrics_names, results))
        else:
            metrics = {"loss": float(results)}

        if "accuracy" in metrics:
            print(f"Test Accuracy (model.evaluate): {metrics['accuracy'] * 100:.4f}%")
        elif "acc" in metrics:
            print(f"Test Accuracy (model.evaluate): {metrics['acc'] * 100:.4f}%")
        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print("model.evaluate failed with current label format.")
        print(f"Reason: {exc}")

    # Always compute accuracy manually for reliability.
    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.astype(int)

    manual_acc = np.mean(y_pred == y_true)
    print(f"Manual Test Accuracy: {manual_acc * 100:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check model accuracy on the disjoint 10k test dataset."
    )
    parser.add_argument(
        "--model",
        default=r"d:\amrita intern\only-accuracy-inc\new_res\global_model_60k_train_1k_test\normal_model_10000.h5",
        help="Path to the trained .h5 model file",
    )
    parser.add_argument(
        "--test",
        default=r"d:\amrita intern\only-accuracy-inc\new_res\mnist_100_clients_60_per_class_1k_test\test_10000_samples_disjoint.npz",
        help="Path to the 10k disjoint test .npz file",
    )
    args = parser.parse_args()

    evaluate_model(args.model, args.test)
