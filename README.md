# SMS Spam Classification — NLP-Based Text Classifier

An end-to-end NLP project that classifies SMS messages as **spam** or **ham (not spam)** using classical text preprocessing, feature engineering, exploratory data analysis, and a comparison of multiple machine learning classifiers.

## Overview

This notebook builds a complete spam-detection pipeline on the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 labeled SMS messages. It walks through data cleaning, exploratory data analysis (EDA), text preprocessing, feature extraction (Bag-of-Words and TF-IDF), and benchmarking 11 different classification algorithms to identify the best-performing model.

## Dataset

- **File:** `spam.csv`
- **Size:** 5,572 messages (5,169 after deduplication)
- **Columns:** `v1` (label: `ham`/`spam`), `v2` (raw message text)
- **Class distribution:** Imbalanced — majority ham, minority spam

## Pipeline

### 1. Data Cleaning
- Dropped unused/empty columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
- Renamed columns to `target` and `text`
- Label-encoded target (`ham` → 0, `spam` → 1)
- Removed 403 duplicate records

### 2. Exploratory Data Analysis (EDA)
- Class balance visualization (bar chart, pie chart)
- Derived features: character count, word count, sentence count per message
- Distribution comparison between spam and ham messages (histograms)
- Correlation heatmap of numeric features
- Word clouds for spam vs. ham vocabulary
- Top-25 most frequent words in each class

### 3. Text Preprocessing
A custom `transform_text()` function applies:
- Lowercasing
- Tokenization (NLTK)
- Removal of non-alphanumeric tokens
- Stopword and punctuation removal
- Stemming (Porter Stemmer)

### 4. Feature Engineering
Two vectorization approaches were tested:
- **CountVectorizer** (Bag-of-Words)
- **TF-IDF Vectorizer** (max_features=3000) — used for final model comparison

### 5. Model Training & Comparison
11 classifiers were trained and evaluated on an 80/20 train-test split:

| Algorithm | Description |
|---|---|
| SVC | Support Vector Classifier (sigmoid kernel) |
| KNN | K-Nearest Neighbors |
| Decision Tree | max_depth=5 |
| Multinomial Naive Bayes | — |
| Logistic Regression | L1 penalty |
| Random Forest | 50 estimators |
| AdaBoost | 50 estimators |
| Bagging Classifier | 50 estimators |
| Extra Trees | 50 estimators |
| Gradient Boosting | 50 estimators |
| XGBoost | — |

## Results

Models ranked by **precision** (prioritized due to class imbalance — minimizing false positives on spam detection):

| Algorithm | Accuracy | Precision |
|---|---|---|
| KNN | 0.905 | **1.000** |
| Naive Bayes | 0.971 | **1.000** |
| Random Forest | 0.976 | 0.983 |
| SVC | 0.976 | 0.975 |
| Extra Trees | 0.975 | 0.975 |
| Logistic Regression | 0.958 | 0.970 |
| AdaBoost | 0.960 | 0.929 |
| XGBoost | 0.967 | 0.926 |
| Gradient Boosting | 0.947 | 0.919 |
| Bagging Classifier | 0.958 | 0.868 |
| Decision Tree | 0.932 | 0.833 |

**Best performing model:** Multinomial Naive Bayes with TF-IDF features achieves a strong balance of high accuracy (97.1%) and perfect precision (1.00), making it well-suited for spam filtering where false positives (flagging legitimate messages as spam) are especially costly.

## Tech Stack

- **Data handling:** `pandas`, `numpy`
- **NLP:** `nltk` (tokenization, stopwords, stemming)
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **ML:** `scikit-learn` (vectorizers, classifiers, metrics), `xgboost`

## Requirements

```
pandas
numpy
nltk
matplotlib
seaborn
wordcloud
scikit-learn
xgboost
```

Also requires the following NLTK downloads:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1. Place `spam.csv` in the project root.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook `spam.ipynb` cell by cell.
4. Review EDA plots, word clouds, and the final `performance_df` comparison table.

## Future Improvements

- Address class imbalance with SMOTE or class-weighting
- Try deep learning approaches (LSTM, BERT/DistilBERT fine-tuning) for richer semantic representation
- Hyperparameter tuning (GridSearchCV/Optuna) on top-performing models
- Persist the best model + vectorizer (`pickle`/`joblib`) and wrap in a simple inference API
- Cross-validation instead of a single train/test split for more robust evaluation
