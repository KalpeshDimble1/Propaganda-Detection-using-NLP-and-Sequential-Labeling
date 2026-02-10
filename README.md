# Propaganda Detection using NLP and Sequential Labeling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive NLP project implementing multiple machine learning approaches for detecting propaganda techniques in text, including traditional ML pipelines, ensemble methods, and deep learning architectures.

## 🎯 Overview

This project tackles two interconnected propaganda detection tasks:

1. **Task 1: Technique Classification** - Identify which propaganda technique is used in a given text span
2. **Task 2: Span Detection & Classification** - Locate propaganda spans in sentences and classify their techniques

The system detects **9 propaganda techniques**:
- Flag-Waving
- Appeal to Fear/Prejudice  
- Causal Oversimplification
- Doubt
- Exaggeration/Minimization
- Loaded Language
- Name Calling/Labeling
- Repetition
- No Propaganda

## ✨ Features

- **Multiple Modeling Approaches**: Traditional ML (SVM, Naive Bayes, Logistic Regression), Ensemble (XGBoost), and Deep Learning (BiLSTM-CRF)
- **Feature Engineering**: TF-IDF vectorization, n-gram analysis, Word2Vec embeddings
- **Hyperparameter Optimization**: GridSearchCV for systematic tuning
- **Sequence Labeling**: BIO tagging schema implementation in PyTorch
- **Comprehensive Evaluation**: Precision, Recall, F1-Score metrics with class imbalance analysis
- **End-to-End Pipeline**: From preprocessing to model deployment

## 📊 Dataset

The project uses annotated propaganda detection datasets with:
- **Training samples**: 1,000+ annotated sentences
- **Validation split**: Stratified split maintaining class distribution
- **Annotation format**: TSV files with BOS/EOS tags marking propaganda spans
- **Label distribution**: Imbalanced with flag_waving as dominant class

## 🧠 Methodology

### Task 1: Technique Classification

**Approach 1: TF-IDF + Classical ML**
- Features: Bag-of-Words (unigrams + bigrams) with TF-IDF weighting
- Models: Logistic Regression, Multinomial Naive Bayes, SVM
- Optimization: GridSearchCV with 3-fold cross-validation

**Approach 2: Word2Vec Embeddings**
- Pre-trained 100D GloVe embeddings
- Sentence-level averaging
- Classifiers: Logistic Regression, SVM

### Task 2: Span Detection & Classification

**Approach 1: BiLSTM-CRF**
- Sequence labeling with BIO tagging schema
- Bidirectional LSTM with CRF layer
- Custom PyTorch implementation with padding handling

**Approach 2: XGBoost Pipeline**
- TF-IDF feature extraction
- Two-stage: span detection (regression) + classification
- Ensemble tree boosting

**Approach 3: Logistic Regression Pipeline**
- Direct span classification
- TF-IDF vectorization
- Lightweight alternative to deep learning

### Preprocessing Pipeline

```python
# Text normalization
- Lowercasing
- BOS/EOS tag removal
- Punctuation and stopword removal
- Stemming (PorterStemmer)
- Lemmatization (WordNetLemmatizer)
- Outlier filtering using IQR
- Custom tokenization and padding for sequence models
```

## 📈 Results

### Task 1: Technique Classification

| Model | Accuracy | Best Class (F1) | Worst Class (F1) |
|-------|----------|-----------------|------------------|
| **Logistic Regression (TF-IDF)** | **40.2%** | flag_waving (0.62) | loaded_language (0.26) |
| SVM (TF-IDF) | 39.4% | flag_waving (0.63) | repetition (0.42) |
| Multinomial NB | 37.9% | flag_waving (0.57) | loaded_language (0.00) |
| Word2Vec + LR | 33.3% | flag_waving (0.60) | loaded_language (0.20) |

### Task 2: Span Detection & Classification

| Approach | Accuracy | Best Class (F1) | Worst Class (F1) |
|----------|----------|-----------------|------------------|
| **Logistic Regression Pipeline** | **33%** | flag_waving (0.59) | loaded_language (0.15) |
| XGBoost Pipeline | 27% | flag_waving (0.49) | name_calling (0.04) |
| BiLSTM-CRF | 58.7%* | O-label (0.74) | All B-/I- tags (0.00) |

*BiLSTM-CRF high accuracy is misleading due to majority class bias

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **PyTorch** - Deep learning framework

### NLP & ML
- **NLTK** - Text preprocessing
- **scikit-learn** - Traditional ML algorithms
- **XGBoost** - Gradient boosting
- **GloVe** - Word embeddings

### Visualization & Analysis
- **Matplotlib** - Plotting
- **Seaborn** - Statistical visualization
- **Classification reports** - Model evaluation

## 🔮 Future Work

### Planned Improvements

1. **Advanced Models**
   - Implement BERT/RoBERTa for contextual embeddings
   - Fine-tune transformer models on propaganda corpus
   - Multi-task learning for joint span+technique prediction

2. **Data Augmentation**
   - Back-translation for minority classes
   - Paraphrasing techniques
   - Synthetic data generation

3. **Architecture Enhancements**
   - Hierarchical attention mechanisms
   - Ensemble of deep learning models
   - Active learning for annotation efficiency

4. **Deployment**
   - REST API for real-time detection
   - Web interface for interactive analysis
   - Browser extension for social media monitoring

5. **Cross-Domain Applications**
   - Fake news detection
   - Bias identification in media
   - Misinformation flagging systems
