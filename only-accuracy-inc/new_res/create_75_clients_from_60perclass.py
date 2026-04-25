import argparse
from pathlib import Path

import numpy as np


def natural_client_sort_key(path: Path) -> int:
    name = path.stem  # client_1
    return int(name.split("_")[1])


def load_clients(dataset_dir: Path, expected_clients: int = 100) -> tuple[np.ndarray, np.ndarray]:
    client_files = sorted(dataset_dir.glob("client_*.npz"), key=natural_client_sort_key)

    if len(client_files) < expected_clients:
        raise ValueError(
            f"Expected at least {expected_clients} client files in {dataset_dir}, found {len(client_files)}."
        )

    # Use exactly the first N clients by numeric order (client_1 ... client_100).
    client_files = client_files[:expected_clients]

    x_parts = []
    y_parts = []

    for file_path in client_files:
        with np.load(file_path) as data:
            if "x" not in data or "y" not in data:
                raise KeyError(f"File {file_path} does not contain 'x' and 'y' arrays.")
            x_parts.append(data["x"])
            y_parts.append(data["y"])

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    if len(x_all) != len(y_all):
        raise ValueError("Combined x and y lengths do not match.")

    return x_all, y_all


def split_and_save(
    x_all: np.ndarray,
    y_all: np.ndarray,
    output_dir: Path,
    num_clients: int = 50,
    seed: int = 42,
) -> None:
    total_samples = len(x_all)
    if total_samples % num_clients != 0:
        raise ValueError(
            f"Total samples ({total_samples}) are not divisible by {num_clients}."
        )

    samples_per_client = total_samples // num_clients

    rng = np.random.default_rng(seed)
    indices = np.arange(total_samples)
    rng.shuffle(indices)

    x_shuffled = x_all[indices]
    y_shuffled = y_all[indices]

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client

        x_client = x_shuffled[start:end]
        y_client = y_shuffled[start:end]

        out_file = output_dir / f"client_{i + 1}.npz"
        np.savez_compressed(out_file, x=x_client, y=y_client)

    print(f"Saved {num_clients} clients to: {output_dir}")
    print(f"Samples per client: {samples_per_client}")
    print(f"Total samples written: {num_clients * samples_per_client}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine 100 client files (60 samples/class setup), then split equally into 50 clients."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_60perclass"),
        help="Path to dataset_60perclass folder containing client_1.npz ... client_100.npz",
    )
    parser.add_argument(
        "--output-subfolder",
        type=str,
        default="clients_50_from_100x60",
        help="Subfolder name to create inside dataset-dir for the new 50-client split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before splitting",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output_dir = dataset_dir / args.output_subfolder

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    x_all, y_all = load_clients(dataset_dir=dataset_dir, expected_clients=100)
    split_and_save(
        x_all=x_all,
        y_all=y_all,
        output_dir=output_dir,
        num_clients=50,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
