# classification-using-pytorch-tutorial
A tutorial on performing classification using PyTorch modules 

# Personality Classification Using PyTorch

A deep learning project that classifies personality types (Extrovert vs Introvert) using behavioral and social data with PyTorch neural networks.

## 📋 Project Overview

This project implements two different neural network architectures to classify personality types based on behavioral indicators. The models compare the effectiveness of ReLU and Sigmoid activation functions in personality prediction tasks.

## 📊 Dataset

**Source**: [Extrovert vs Introvert Personality Traits Dataset](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data) from Kaggle

**Description**: A rich collection of behavioral and social data designed to explore the spectrum of human personality, capturing key indicators of extroversion and introversion.

### Features
- `Time_spent_Alone`: Hours spent alone per day
- `Stage_fear`: Presence of stage fear (Yes/No)
- `Social_event_attendance`: Frequency of social event attendance
- `Going_outside`: Frequency of going outside
- `Drained_after_socializing`: Whether socializing drains energy (Yes/No)
- `Friends_circle_size`: Number of close friends
- `Post_frequency`: Social media posting frequency

### Target Variable
- `Personality`: Extrovert (0) or Introvert (1)

### Dataset Statistics
- **Total Records**: 2,900 entries
- **After Preprocessing**: 2,585 entries (removed ~11% with missing values)
- **Class Distribution**: Fairly balanced between Extrovert and Introvert

## 🏗️ Model Architectures

### Model 1: ReLU Activation
```
Input Layer (7 features) → Hidden Layer (40 neurons) → Hidden Layer (20 neurons) → Output Layer (2 classes)
Activation Functions: ReLU for hidden layers
```

### Model 2: Sigmoid Activation
```
Input Layer (7 features) → Hidden Layer (50 neurons) → Hidden Layer (25 neurons) → Output Layer (2 classes)
Activation Functions: Sigmoid for hidden layers
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch numpy pandas matplotlib scikit-learn
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/personality-classification-pytorch.git
cd classification-using-pytorch-tutorial
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data) and place `personality_dataset.csv` in the project directory.

3. Run the Jupyter notebook:
```bash
jupyter notebook classification-using-pytorch-tutorial.ipynb
```

## 📈 Results

### Performance Metrics

| Metric | Model 1 (ReLU) | Model 2 (Sigmoid) |
|--------|----------------|-------------------|
| Accuracy | 94.78% | 94.78% |
| Precision | 94.51% | 94.51% |
| Recall | 94.88% | 94.88% |
| F1-Score | 94.70% | 94.70% |
| AUC | 0.98 | 0.93 |

### Confusion Matrix (Both Models)
```
[[249  14]
 [ 13 241]]
```

## 🔍 Key Insights

1. **Performance**: Both models achieved identical accuracy, precision, recall, and F1-scores, indicating robust performance across different architectures.

2. **Activation Function Comparison**: While most metrics were identical, Model 1 (ReLU) showed superior discriminatory power with a higher AUC score (0.98 vs 0.93).

3. **Model Recommendation**: Model 1 with ReLU activation is recommended due to:
   - Higher AUC score indicating better classification confidence
   - ReLU's advantage in preventing vanishing gradient problems
   - Better convergence during training

## 📁 Project Structure

```
personality-classification-pytorch/
│
├── classification-using-pytorch-tutorial.ipynb    # Main notebook
├── personality_dataset.csv                  # Dataset file
├── README.md                                # Project documentation
└── requirements.txt                         # Dependencies
```

## 🛠️ Data Preprocessing

1. **Label Encoding**: Converted categorical variables (`Stage_fear`, `Drained_after_socializing`, `Personality`) to numerical format
2. **Missing Value Handling**: Removed rows with NaN values (~11% of dataset)
3. **Feature Selection**: Used all behavioral indicators as input features
4. **Train-Test Split**: 80% training, 20% testing with stratification

## 📚 Technical Details

- **Framework**: PyTorch
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate: 1e-6)
- **Training Epochs**: 100,000
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Future Improvements

- Implement additional activation functions (Tanh, Leaky ReLU)
- Add regularization techniques (Dropout, Batch Normalization)
- Experiment with different network architectures
- Implement cross-validation for more robust evaluation
- Add feature importance analysis
- Create a web interface for personality prediction

## 👤 Author

**Ebube Ndubuisi**
- Course: ANLY-6500: Advanced Data Analytics
- Focus: Classification Using PyTorch

## 🙏 Acknowledgments

- Dataset provided by Rakesh Kapilavai on Kaggle
- PyTorch documentation and community
- Scikit-learn for preprocessing and evaluation metrics

---

*This project was developed as part of a machine learning course assignment, demonstrating the application of deep learning techniques to personality classification problems.*
