# Facial Expression Recognition Challenge

This repository contains my implementation for the [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge) on Kaggle. The project focuses on building and evaluating deep learning models to classify facial expressions into seven different emotion categories.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Setup and Installation](#-setup-and-installation)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Experiments and Results](#-experiments-and-results)
- [Weights & Biases Integration](#-weights--biases-integration)
- [Findings and Analysis](#-findings-and-analysis)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Project Overview

This project explores various neural network architectures for facial expression recognition, with a focus on understanding how different hyperparameters and architectural choices affect model performance. The implementation is done using PyTorch, and all experiments are tracked using Weights & Biases (WandB) for comprehensive analysis and visualization.

## 📊 Dataset

The dataset consists of 48x48 pixel grayscale images of faces, each labeled with one of seven emotion categories:
- 0: Angry
- 1: Disgust
- 2: Fear
- 3: Happy
- 4: Sad
- 5: Surprise
- 6: Neutral

## 🛠️ Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Facial-Expression-Recognition.git
   cd Facial-Expression-Recognition
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Weights & Biases:
   ```bash
   wandb login
   ```

## 📁 Project Structure

```
Facial-Expression-Recognition/
├── data/                    # Dataset storage
├── models/                  # Model architectures
│   ├── cnn.py
│   ├── resnet.py
│   └── ...
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── configs/                 # Configuration files
├── requirements.txt
└── README.md
```

## 🧠 Methodology

The project follows an iterative approach to model development:

1. **Baseline Model**: Start with a simple CNN architecture
2. **Progressive Complexity**: Gradually increase model complexity
3. **Regularization**: Implement various techniques to prevent overfitting
4. **Transfer Learning**: Utilize pre-trained models
5. **Ensemble Methods**: Combine multiple models for improved performance

## 📈 Experiments and Results

All experiments are tracked using Weights & Biases. Key metrics logged include:
- Training/Validation loss and accuracy
- Learning rate schedules
- Model hyperparameters
- Confusion matrices
- Sample predictions

### Experimented Architectures
1. **Simple CNN**
   - 3-4 Convolutional layers
   - MaxPooling and BatchNorm
   - Dropout for regularization

2. **ResNet Variants**
   - ResNet18
   - ResNet34
   - With/without pre-trained weights

3. **EfficientNet**
   - Different scaling variants
   - Fine-tuning strategies

## 🔍 Weights & Biases Integration

All experiments are logged to Weights & Biases with the following structure:
- Project: `facial-expression-recognition`
- Tags: `[model_type, dataset_version, experiment_type]`
- Config: Hyperparameters and model architecture
- Metrics: Training/validation metrics
- Artifacts: Model checkpoints

## 📝 Findings and Analysis

Key insights from the experiments:
1. **Overfitting Challenges**: Addressed using data augmentation and dropout
2. **Class Imbalance**: Explored weighted loss functions
3. **Learning Rate Scheduling**: Impact on model convergence
4. **Model Depth**: Trade-offs between complexity and performance

## 🚀 Usage

### Training a Model
```bash
python src/train.py --model cnn --epochs 50 --batch_size 64 --lr 0.001
```

### Evaluation
```bash
python src/evaluate.py --model_path models/best_model.pth
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
