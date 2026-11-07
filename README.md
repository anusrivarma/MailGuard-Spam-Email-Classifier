# MailGuard — Spam Email Classifier

A beginner-friendly machine-learning project that classifies email text as **Spam** or **Not Spam (Ham)**. It includes a polished Streamlit dashboard.

## Features

- Professional MailGuard dashboard
- Spam / Not Spam predictions
- Model confidence display (when supported by the model)
- Accuracy, precision, recall and F1-score cards
- Confusion matrix
- Sample email messages for quick testing

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF text feature extraction
- Multinomial Naive Bayes
- Streamlit

## Project structure

```text
MailGuard_Professional_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── RUN_MAILGUARD.bat
├── .gitignore
└── data/
    └── emails.csv
```

The trained model (`spam_classifier.joblib`) and evaluation file (`metrics.json`) are generated when you run `train_model.py`.

## Run on Windows (Antigravity or VS Code)

1. Extract this ZIP file.
2. Open the extracted `MailGuard_Professional_Project` folder in Antigravity or VS Code.
3. Open **Terminal → New Terminal**.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Train the model:

   ```bash
   python train_model.py
   ```

6. Start the web app:

   ```bash
   python -m streamlit run app.py
   ```

7. Open the local browser page shown in the terminal.

### Optional one-click launcher

On Windows, double-click `RUN_MAILGUARD.bat`. It installs the dependencies, trains the model, and starts the Streamlit app. Keep the terminal window open while using the app.

## How the machine-learning pipeline works

1. The dataset contains email text and a label (`spam` or `ham`).
2. TF-IDF converts words and two-word phrases into numerical features.
3. Multinomial Naive Bayes learns patterns from the labelled examples.
4. The trained pipeline predicts the category of a new email.
5. The training script calculates accuracy, precision, recall, F1-score and a confusion matrix on a held-out test set.

## Dataset limitation

The included `data/emails.csv` is a small, hand-prepared educational sample dataset. Its evaluation results are for demonstration and learning only; they do **not** establish production-level spam detection. For a stronger portfolio project, use a larger public dataset, record its source, retrain, and report the resulting test metrics honestly.

## Troubleshooting

- If `python` is not recognized, install Python and enable **Add Python to PATH**.
- If a package is missing, run `python -m pip install -r requirements.txt`.
- If the app says the model is missing, run `python train_model.py` first.
- If PowerShell blocks virtual-environment activation, you can use Command Prompt or run the commands without activating the environment.
- If Streamlit is not recognized, use `python -m streamlit run app.py`.

## Resume description

**MailGuard — Spam Email Classifier**
- Built a Python machine-learning application to classify email messages as spam or legitimate.
- Used TF-IDF for text feature extraction and Multinomial Naive Bayes for classification.
- Evaluated the model using accuracy, precision, recall, F1-score and a confusion matrix.
- Developed an interactive Streamlit dashboard for email testing.

Only include model performance numbers on your resume after retraining and checking the generated `metrics.json`.

## Application Preview

### Spam Email Detection
![Spam Detection](/screenshots/Spam-email.png)

### Legitimate Email Detection
![Legitimate Email](/screenshots/Normal-email.png)
