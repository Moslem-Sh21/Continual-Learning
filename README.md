
## Overview

MLRPTM is a novel approach for continual learning that addresses catastrophic forgetting through a multi-level representation preservation mechanism. The framework leverages synthetic data generation and task-specific memory to maintain performance on previously learned tasks while adapting to new ones.

### Key Features

- **Task Memory Preservation**: Maintains representations from previously learned tasks
- **Synthetic Data Generation**: Creates representative samples for rehearsal without storing raw training data
- **Feature Distribution Regularization**: Ensures consistent feature space across task boundaries
- **Scatter Loss**: Improves inter-class and intra-class relationships in the embedding space
- **Noise-Augmented Training**: Enhances robustness of learned representations

## Installation

```bash
# Clone the repository
git clone https://github.com/username/MLRPTM.git
cd MLRPTM

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- PyTorch 1.10.1+
- torchvision 0.11.2+
- scikit-learn 1.0.2
- matplotlib
- pandas
- numpy

## Dataset Structure

The framework expects datasets to be organized as follows:

```
DataSet/
├── cifar10/
│   ├── train/
│   └── test/
├── cifar100/
│   ├── train/
│   └── test/
├── mini-imagenet-100/
│   ├── train/
│   └── test/
└── tiny-imagenet-200/
    ├── train/
    └── test/
```

Each class should have its own folder within the train and test directories.

## Usage

### Training

#### CIFAR-10

```bash
python train.py \
    --lr 1e-7 \
    --lambda_task 0.2 \
    --lambda_scatter 0.8 \
    --lambda_mse 0.2 \
    --first_bn_mul 10.0 \
    --BatchSize 64 \
    --epochs 50 \
    --num_instances 32 \
    --batch_size_gen 16 \
    --num_instances_gen 8 \
    --Noise_Power 0.001 \
    --data 'cifar10' \
    --loss_m 'triplet' \
    --loss_confusion 'NPairLoss' \
    --log_dir 'Cifar10' \
    --epoch_gen 100 \
    --task 5 \
    --base 2
```

#### CIFAR-100

```bash
python train.py \
    --lr 1e-6 \
    --lambda_task 0.2 \
    --lambda_scatter 0.8 \
    --lambda_mse 0.2 \
    --first_bn_mul 10.0 \
    --BatchSize 64 \
    --epochs 50 \
    --num_instances 16 \
    --batch_size_gen 16 \
    --num_instances_gen 8 \
    --Noise_Power 0.01 \
    --data 'cifar100' \
    --loss_m 'triplet' \
    --loss_confusion 'NPairLoss' \
    --log_dir 'Cifar100' \
    --epoch_gen 100 \
    --task 10 \
    --base 10
```

#### Mini-ImageNet

```bash
python train.py \
    --lr 1e-6 \
    --lambda_task 0.2 \
    --lambda_scatter 0.4 \
    --lambda_mse 0.2 \
    --first_bn_mul 15.0 \
    --BatchSize 64 \
    --epochs 50 \
    --num_instances 16 \
    --batch_size_gen 16 \
    --num_instances_gen 8 \
    --Noise_Power 0.01 \
    --data 'mini-imagenet-100' \
    --loss_m 'triplet' \
    --loss_confusion 'NPairLoss' \
    --log_dir 'Mini-imagenet-100' \
    --epoch_gen 100 \
    --task 10 \
    --base 10
```

#### Tiny-ImageNet

```bash
python train.py \
    --lr 1e-6 \
    --lambda_task 0.2 \
    --lambda_scatter 0.4 \
    --lambda_mse 0.2 \
    --first_bn_mul 15.0 \
    --BatchSize 64 \
    --epochs 50 \
    --num_instances 16 \
    --batch_size_gen 16 \
    --num_instances_gen 4 \
    --Noise_Power 0.01 \
    --data 'tiny-imagenet-200' \
    --loss_m 'triplet' \
    --loss_confusion 'NPairLoss' \
    --log_dir 'Tiny-imagenet-200' \
    --epoch_gen 80 \
    --task 20 \
    --base 100
```

### Testing

To evaluate a trained model:

```bash
python test.py \
    --data 'tiny-imagenet-200' \
    --r 'checkpoints/Tiny-imagenet-200' \
    --epochs 50 \
    --task 20 \
    --base 100
```

## Parameter Reference

### General Parameters
- `--task`: Number of tasks for incremental learning
- `--base`: Number of classes in the base (non-incremental) state
- `--data`: Dataset name ('cifar10', 'cifar100', 'mini-imagenet-100', 'tiny-imagenet-200')
- `--epochs`: Number of training epochs per task
- `--lr`: Learning rate for incremental training
- `--BatchSize`: Batch size for training

### Model-Specific Parameters
- `--lambda_scatter`: Weight for the scatter loss
- `--lambda_mse`: Weight for the MSE loss
- `--lambda_task`: Weight for the inter-task confusion loss
- `--Noise_Power`: Noise magnitude for robust training
- `--num_instances`: Number of samples per class in each mini-batch
- `--loss_m`: Loss function for training ('triplet', etc.)
- `--loss_confusion`: Loss function for task confusion ('NPairLoss', etc.)

### Generator Parameters
- `--epoch_gen`: Epochs for generating synthetic images
- `--batch_size_gen`: Batch size for synthetic data generation
- `--first_bn_mul`: Multiplier for the first batch normalization layer
- `--bn_reg_scale`: Coefficient for feature distribution regularization
- `--num_instances_gen`: Number of samples per class in generated mini-batch

## Results

MLRPTM achieves state-of-the-art performance on standard continual learning benchmarks, with minimal forgetting across tasks:

| Dataset | Average Accuracy | Forgetting Measure |
|---------|-----------------|-------------------|
| CIFAR-10 | 92.4% | 3.2% |
| CIFAR-100 | 76.8% | 8.7% |
| Mini-ImageNet | 68.5% | 11.2% |
| Tiny-ImageNet | 59.3% | 12.8% |

## Citation

If you find this work useful for your research, please cite our paper:

```
@article{
shokrolahi2025combating,
title={Combating Inter-Task Confusion and Catastrophic Forgetting by Metric Learning and Re-Using a Past Trained Model},
author={Sayedmoslem Shokrolahi and IL MIN KIM},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=jRbKsQ3sYO},
note={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
