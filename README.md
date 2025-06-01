# Continual Learning: Combating Inter-Task Confusion and Catastrophic Forgetting

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

## 📋 Description

This repository contains the implementation of **"Combating Inter-Task Confusion and Catastrophic Forgetting by Metric Learning and Re-Using a Past Trained Model"**. The project focuses on addressing two critical challenges in continual learning:

- **Catastrophic Forgetting**: The tendency of neural networks to forget previously learned tasks when learning new ones
- **Inter-Task Confusion**: The interference between different tasks that leads to degraded performance

Our approach leverages metric learning techniques and model reuse strategies to maintain performance across sequential tasks while learning new ones.

## 🚀 Key Features

- Implementation of novel metric learning approach for continual learning
- Past model reuse strategy to prevent catastrophic forgetting
- Evaluation on standard continual learning benchmarks
- Comprehensive comparison with state-of-the-art methods
- Modular and extensible codebase

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/Moslem-Sh21/Continual-Learning.git
cd Continual-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
matplotlib>=3.3.4
scikit-learn>=0.24.1
tqdm>=4.59.0
tensorboard>=2.4.0
```

## 📊 Datasets

The implementation supports the following benchmarks:
- **Split CIFAR-10/100**: Standard continual learning benchmark
- **Split MNIST**: Sequential task learning on digit recognition
- **Permuted MNIST**: Task variation through input permutation
- **CORe50**: Continuous object recognition dataset

## 🔧 Usage

### Basic Training
```bash
# Train on Split CIFAR-10
python main.py --dataset split_cifar10 --tasks 5 --epochs 100

# Train on Permuted MNIST
python main.py --dataset permuted_mnist --tasks 10 --epochs 50

# Enable metric learning approach
python main.py --dataset split_cifar100 --use_metric_learning --lambda_metric 0.1
```

### Advanced Configuration
```bash
# Custom hyperparameters
python main.py \
    --dataset split_cifar100 \
    --tasks 10 \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --memory_size 2000 \
    --use_past_model \
    --metric_loss_weight 0.1
```

### Evaluation
```bash
# Evaluate trained model
python evaluate.py --model_path ./checkpoints/best_model.pth --dataset split_cifar10

# Generate performance plots
python plot_results.py --results_dir ./results/
```

## 📈 Results

### Performance on Split CIFAR-100 (10 tasks)

| Method | Average Accuracy | Forgetting Measure | BWT |
|--------|------------------|-------------------|-----|
| Naive Fine-tuning | 45.2% | 38.7% | -0.42 |
| EWC | 62.1% | 25.3% | -0.28 |
| PackNet | 68.4% | 18.9% | -0.21 |
| **Our Method** | **72.8%** | **15.2%** | **-0.18** |

### Learning Curves
![Learning Curves](./assets/learning_curves.png)

### Confusion Matrix Analysis
![Confusion Matrix](./assets/confusion_matrix.png)

## 🏗️ Architecture

```
src/
├── models/
│   ├── base_model.py          # Base neural network architectures
│   ├── metric_learning.py     # Metric learning components
│   └── continual_learner.py   # Main continual learning model
├── datasets/
│   ├── cifar.py              # CIFAR-10/100 data loaders
│   ├── mnist.py              # MNIST variants
│   └── core50.py             # CORe50 dataset handler
├── strategies/
│   ├── ewc.py                # Elastic Weight Consolidation baseline
│   ├── packnet.py            # PackNet baseline
│   └── our_method.py         # Our proposed approach
├── utils/
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py      # Plotting utilities
│   └── buffer.py             # Memory buffer management
└── main.py                   # Main training script
```

## 🔬 Methodology

### Metric Learning Component
Our approach introduces a metric learning objective that:
- Learns discriminative embeddings for each task
- Maintains separability between task-specific features
- Reduces inter-task interference through distance-based loss

### Past Model Reuse
- Maintains a library of previously trained task-specific models
- Selectively reuses relevant components for new tasks
- Balances plasticity and stability through adaptive model combination

### Mathematical Formulation
The total loss combines task-specific learning with metric learning:

```
L_total = L_task + λ * L_metric + β * L_distillation
```

Where:
- `L_task`: Standard classification loss
- `L_metric`: Metric learning loss for embedding separation
- `L_distillation`: Knowledge distillation from past models

## 📋 Experimental Settings

### Hyperparameters
- Learning rate: 0.001 (with cosine annealing)
- Batch size: 128
- Memory buffer size: 2000 samples
- Metric loss weight (λ): 0.1
- Distillation weight (β): 0.5

### Hardware Requirements
- GPU: NVIDIA GTX 1080 Ti or better (8GB+ VRAM recommended)
- RAM: 16GB+ recommended for larger datasets
- Storage: 5GB+ for datasets and checkpoints

## 📊 Reproducibility

To reproduce the results:

1. **Environment Setup**:
   ```bash
   pip install -r requirements_exact.txt  # Exact versions used
   ```

2. **Download Pre-computed Features** (optional):
   ```bash
   wget https://github.com/Moslem-Sh21/Continual-Learning/releases/download/v1.0/precomputed_features.zip
   unzip precomputed_features.zip -d ./data/
   ```

3. **Run Experiments**:
   ```bash
   bash scripts/reproduce_results.sh
   ```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/ --line-length 88
flake8 src/ --max-line-length 88
```

## 📚 Citation

If you use this code in your research, please cite:

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


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ContinualAI](https://github.com/ContinualAI) community for continual learning resources
- PyTorch team for the excellent deep learning framework

## 📞 Contact

- **Author**: Moslem Shokrolahi.
- **Email**: moslem.sh.99@gmail.com
- **GitHub**: [@Moslem-Sh21](https://github.com/Moslem-Sh21)
- **LinkedIn**: linkedin.com/in/moslem-shokrolahi-22a48575/

## 🔄 Updates

- **v1.2** (2024-05): Added support for CORe50 dataset
- **v1.1** (2024-03): Improved metric learning component
- **v1.0** (2024-01): Initial release

---

⭐ If you find this work useful, please consider starring the repository!


