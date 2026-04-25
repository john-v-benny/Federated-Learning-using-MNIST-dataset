import os
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist


def _pack_samples(x, y):
    """Pack (x, y) rows into a sortable representation for set-like comparisons."""
    x_flat = x.reshape(x.shape[0], -1).astype(np.uint16)
    y_col = y.reshape(-1, 1).astype(np.uint16)
    packed = np.concatenate([x_flat, y_col], axis=1)
    return np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.dtype.itemsize * packed.shape[1]))
    ).ravel()


def same_sample_multiset(x1, y1, x2, y2):
    """True if two datasets contain exactly the same samples (ignoring order)."""
    if x1.shape[0] != x2.shape[0]:
        return False
    p1 = np.sort(_pack_samples(x1, y1))
    p2 = np.sort(_pack_samples(x2, y2))
    return np.array_equal(p1, p2)


def intersection_count(x1, y1, x2, y2):
    """Count overlapping samples between two datasets."""
    p1 = _pack_samples(x1, y1)
    p2 = _pack_samples(x2, y2)
    return len(np.intersect1d(p1, p2))


def main():
    base_dir = r"d:\amrita intern\only-accuracy-inc\new_res\mnist_100_clients_60_per_class_1k_test"
    train_file = os.path.join(base_dir, "complete_60000_samples.npz")
    test_1k_file = os.path.join(base_dir, "test_1000_samples.npz")
    out_10k_file = os.path.join(base_dir, "test_10000_samples_disjoint.npz")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Missing training file: {train_file}")
    if not os.path.exists(test_1k_file):
        raise FileNotFoundError(f"Missing 1k test file: {test_1k_file}")

    print("Loading existing saved datasets...")
    train_npz = np.load(train_file)
    test_1k_npz = np.load(test_1k_file)

    x_train_saved = train_npz["x"]
    y_train_saved = train_npz["y"]
    x_test_1k_saved = test_1k_npz["x"]
    y_test_1k_saved = test_1k_npz["y"]

    print(f"Saved train shape: {x_train_saved.shape}, labels: {y_train_saved.shape}")
    print(f"Saved 1k test shape: {x_test_1k_saved.shape}, labels: {y_test_1k_saved.shape}")

    # Reconstruct the original 70k source and the deterministic class-wise split
    # used in your split notebook (SEED=42).
    print("\nReconstructing original 70k source split logic...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    all_x = np.concatenate([x_train, x_test], axis=0)
    all_y = np.concatenate([y_train, y_test], axis=0)

    seed = 42
    all_x, all_y = shuffle(all_x, all_y, random_state=seed)

    num_classes = 10
    train_per_class = 6000
    first_test_per_class = 100
    x_train_ref_parts, y_train_ref_parts = [], []
    x_test_1k_ref_parts, y_test_1k_ref_parts = [], []
    remaining_idx_parts = []

    for c in range(num_classes):
        idx = np.where(all_y == c)[0]
        if len(idx) < train_per_class + first_test_per_class:
            raise ValueError(f"Class {c} has insufficient samples: {len(idx)}")

        train_idx = idx[:train_per_class]
        test_1k_idx = idx[train_per_class:train_per_class + first_test_per_class]
        remaining_idx = idx[train_per_class + first_test_per_class:]

        x_train_ref_parts.append(all_x[train_idx])
        y_train_ref_parts.append(all_y[train_idx])
        x_test_1k_ref_parts.append(all_x[test_1k_idx])
        y_test_1k_ref_parts.append(all_y[test_1k_idx])
        remaining_idx_parts.append(remaining_idx)

    x_train_ref = np.concatenate(x_train_ref_parts, axis=0)
    y_train_ref = np.concatenate(y_train_ref_parts, axis=0)
    x_test_1k_ref = np.concatenate(x_test_1k_ref_parts, axis=0)
    y_test_1k_ref = np.concatenate(y_test_1k_ref_parts, axis=0)
    remaining_idx_all = np.concatenate(remaining_idx_parts)
    if len(remaining_idx_all) != 9000:
        raise RuntimeError(f"Expected 9000 remaining samples, found {len(remaining_idx_all)}")

    x_test_9k = all_x[remaining_idx_all]
    y_test_9k = all_y[remaining_idx_all]

    print("Validating saved files against reconstructed split...")
    train_ok = same_sample_multiset(x_train_saved, y_train_saved, x_train_ref, y_train_ref)
    test1k_ok = same_sample_multiset(x_test_1k_saved, y_test_1k_saved, x_test_1k_ref, y_test_1k_ref)

    if not train_ok:
        raise RuntimeError(
            "Saved 60k training set does not match reconstructed split. "
            "Stopping to avoid creating an incorrect 10k test set."
        )
    if not test1k_ok:
        raise RuntimeError(
            "Saved 1k test set does not match reconstructed split. "
            "Stopping to avoid creating an incorrect 10k test set."
        )

    print("Validation passed.")

    # Build final disjoint 10k test set: existing 1k + remaining disjoint 9k
    x_test_10k = np.concatenate([x_test_1k_saved, x_test_9k], axis=0)
    y_test_10k = np.concatenate([y_test_1k_saved, y_test_9k], axis=0)
    x_test_10k, y_test_10k = shuffle(x_test_10k, y_test_10k, random_state=seed)

    overlap_with_train = intersection_count(x_train_saved, y_train_saved, x_test_10k, y_test_10k)
    if overlap_with_train != 0:
        raise RuntimeError(f"Overlap detected between train and 10k test: {overlap_with_train}")

    np.savez(out_10k_file, x=x_test_10k, y=y_test_10k)

    print("\nDone.")
    print(f"Created disjoint 10k test file: {out_10k_file}")
    print(f"10k test shape: {x_test_10k.shape}, labels: {y_test_10k.shape}")
    print(f"Train-test overlap count: {overlap_with_train}")


if __name__ == "__main__":
    main()
