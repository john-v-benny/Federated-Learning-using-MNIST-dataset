# Weighted Loss-Based Federated Learning

This directory implements a novel weighted federated learning approach based on loss aggregation rather than weight averaging.

## Approach Overview

### Traditional FedAvg (Your existing implementations)
- Clients train locally and return **trained weights**
- Server averages weights: `W_avg = (W1 + W2 + ... + Wn) / n`
- All clients contribute equally

### Weighted Loss-Based FL (This implementation)
- Server assigns importance coefficients to each client: K1, K2, ..., K5
- Server sends **scaled initial weights** to clients: `Ki × W0`
- Clients train locally and return their **losses**: `Li`
- Server aggregates using weighted sum: `L = ΣKi·Li`
- Server updates global model based on aggregated loss: `W1 = F(L)`

## Weight Coefficients

As per the diagram:
- K1 = 0.15 (Client 1)
- K2 = 0.97 (Client 2) - Highest weight
- K3 = 0.08 (Client 3) - Lowest weight
- K4 = 0.2 (Client 4)
- K5 = 0.5 (Client 5)
- **Sum: ΣKi = 1.9 ≥ 1** (satisfies constraint)

## Benefits

1. **Client importance control**: Server can weight reliable clients higher
2. **Handle non-IID data**: Reduce impact of clients with poor data quality
3. **Dynamic adaptation**: K values can be adjusted based on performance
4. **Better convergence**: Prioritize well-performing clients

## Files

### 1. `split_data.ipynb`
Splits MNIST dataset into 5 client partitions:
- Uses stratified K-fold to maintain class distribution
- Saves to `mnist_split_data_new/mnist_part{1-5}.npz`
- Each partition contains train/test splits

### 2. `weighted_federated_learning.ipynb`
Main implementation:
- Implements weighted loss aggregation algorithm
- Trains for 10 rounds with 5 local epochs per round
- Tracks per-client and global metrics
- Generates comprehensive visualizations
- Saves models and results

## Usage

### Step 1: Split the data
```bash
# Run split_data.ipynb to create client partitions
```

### Step 2: Run weighted FL training
```bash
# Run weighted_federated_learning.ipynb
```

## Results

Results are saved to:
- `saved_models/` - Model checkpoints and final model
- `results/` - Training history, plots, and metrics

## Model Architecture

Standard 3-layer neural network used across all workspace experiments:
- Dense(128) + BatchNorm + Dropout(0.3)
- Dense(64) + BatchNorm + Dropout(0.2)
- Dense(32) + BatchNorm + Dropout(0.2)
- Dense(10, softmax)

## Configuration

Key parameters (can be modified in the notebook):
- `NUM_CLIENTS = 5`
- `NUM_ROUNDS = 10`
- `LOCAL_EPOCHS = 5`
- `BATCH_SIZE = 64`
- `LEARNING_RATE = 0.0005`
- `K_COEFFICIENTS = [0.15, 0.97, 0.08, 0.2, 0.5]`

## Comparison with Existing Work

Compare with:
- `5 split data/averaged_training/` - Simple weight averaging
- `5 split data/normal_training/` - Independent training
- Results should show impact of weighted client contributions
