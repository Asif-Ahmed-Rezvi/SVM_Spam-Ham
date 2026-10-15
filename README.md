# SMS Spam Classification Using SVM

A machine-learning project that classifies SMS messages as **spam** or **ham** (legitimate messages) using TF-IDF text features and a Linear Support Vector Machine (SVM).

## Project Overview

The notebook follows a standard text-classification workflow:

1. Load the SMS Spam Collection dataset.
2. Inspect the dataset and explore message lengths and word distributions.
3. Clean and preprocess SMS text.
4. Split the data into training and test sets using a stratified split.
5. Train a baseline SVM model.
6. Tune the TF-IDF and SVM hyperparameters with cross-validation.
7. Evaluate the tuned model and display a confusion matrix.
8. Predict whether custom SMS messages are spam or ham.

## Dataset

- **Dataset:** SMS Spam Collection
- **Download:** [Kaggle — SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Expected CSV filename in the notebook:** `SMS.csv`
- **Expected columns:** `v1` (label) and `v2` (message)

The notebook keeps these two columns and renames them to `label` and `message`. Labels are converted to numeric values during preprocessing: `ham = 0` and `spam = 1`.

Download the dataset and upload the CSV file when prompted by Google Colab. Make sure the uploaded filename matches the filename used in the notebook (`SMS.csv`).

## Technologies and Libraries

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- NLTK
- WordCloud
- scikit-learn
- Google Colab

## Text Preprocessing

The notebook's `clean_text()` function performs these operations:

- Converts text to lowercase.
- Removes URLs and email addresses.
- Removes numeric digits and punctuation.
- Removes English stopwords.
- Lemmatizes words using NLTK's WordNet lemmatizer.
- Removes duplicate dataset rows before modeling.

## Model and Feature Extraction

The project uses a scikit-learn `Pipeline` with:

- **TF-IDF Vectorizer:** Converts text into numerical features. The baseline configuration uses up to 5,000 features and unigrams plus bigrams.
- **LinearSVC:** Classifies the TF-IDF features. The baseline model uses `class_weight='balanced'` to account for the difference in the number of spam and ham messages.

The notebook then uses `GridSearchCV` with five-fold cross-validation and **F1-score** as the tuning objective. The search covers TF-IDF vocabulary size, n-gram range, minimum document frequency, SVM regularization (`C`), loss function, and class weighting.

## Baseline vs. Tuned SVM Results

The results report associated with this project records the following test-set metrics:

| Metric | Baseline SVM | Tuned SVM |
|---|---:|---:|
| Accuracy | 97.8% | 98.5% |
| Spam precision | 92.2% | 98.3% |
| Spam recall | 90.1% | 90.1% |
| Spam F1-score | 91.1% | 94.0% |

### Interpretation

- **Accuracy** increased from 97.8% to 98.5%.
- **Spam precision** increased from 92.2% to 98.3%, meaning fewer legitimate messages were incorrectly flagged as spam.
- **Spam recall** remained at 90.1%, so the tuned model did not improve the proportion of spam messages detected.
- **Spam F1-score** increased from 91.1% to 94.0%, reflecting a better balance between precision and recall.

**Evaluation note:** Treat this as a direct comparison only if both models were evaluated on the same held-out test split. Re-run the notebook and update the figures if your current run produces different results.

## Evaluation

The notebook reports accuracy, precision, recall, F1-score, and a classification report. It also plots a confusion matrix for the tuned model.

- **False positive:** A legitimate message predicted as spam.
- **False negative:** A spam message predicted as legitimate.

Inspecting these errors helps reveal how the model behaves beyond its overall accuracy.

## Try a Custom Message

After the notebook trains the tuned model, its `predict_sms()` function can classify a new message:

```python
print(predict_sms("Congratulations! You have won a free iPhone. Click here now."))
print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))
```

The function returns either `Spam` or `Ham`. These are model predictions, not guarantees.

## How to Run in Google Colab

1. Open `SMS_Spam_SVM_Modified.ipynb` in Google Colab.
2. Download the SMS Spam Collection dataset from the Kaggle link above.
3. Run the installation and import cells.
4. Upload the dataset file when prompted. Ensure its name matches `SMS.csv`.
5. Run the notebook cells in order from top to bottom.
6. Review the evaluation metrics, confusion matrix, and sample predictions.

The notebook installs the required libraries, including `wordcloud`, `nltk`, and `scikit-learn`. NLTK stopwords and WordNet resources are downloaded in the notebook.

## Limitations

- The dataset is a collection of SMS messages and may not represent current spam patterns or other messaging platforms.
- The preprocessing is English-focused and may not work well for other languages or mixed-language messages.
- Removing numbers, URLs, punctuation, and stopwords can remove useful signals in some messages.
- A high test score does not guarantee reliable performance on messages outside this dataset.
- The notebook's cross-validation and final evaluation should be interpreted according to how the data split and model selection are performed; keep the test set separate from model selection for an unbiased final estimate.

## Project Structure

```text
.
├── SMS_Spam_SVM_Modified.ipynb
├── SMS_Spam_SVM_Results.md
└── README.md
```

The dataset file (`SMS.csv`) is downloaded separately and uploaded to Colab; it is not included in this project structure by default.

## Future Improvements

- Inspect and document false-positive and false-negative examples.
- Compare additional models using the same train/test split.
- Evaluate precision-recall trade-offs for different decision thresholds.
- Improve validation by keeping model selection separate from the final test evaluation.
- Test on a newer, independent SMS dataset.

## Acknowledgements

Dataset: [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
