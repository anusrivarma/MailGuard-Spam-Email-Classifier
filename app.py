from pathlib import Path
import json

import joblib
import streamlit as st

# ---------------------------------------------------------
# MailGuard - Spam Email Classifier
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_classifier.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(
    page_title="MailGuard | Spam Email Classifier",
    page_icon="🛡️",
    layout="wide",
)

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fc;
    }
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #101b3d, #294b83, #26748a);
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        font-size: 2.35rem;
        margin-bottom: 6px;
    }
    .hero p {
        color: #e2edff;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e3eaf4;
        border-radius: 14px;
        padding: 17px;
        min-height: 100px;
    }
    .metric-name {
        color: #64748b;
        font-size: 0.88rem;
    }
    .metric-number {
        color: #172554;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .spam-result {
        background: #fff1f0;
        border: 1px solid #fecaca;
        color: #991b1b;
        border-radius: 14px;
        padding: 18px;
    }
    .ham-result {
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
        border-radius: 14px;
        padding: 18px;
    }
    .footnote {
        color: #64748b;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path):
    """Load the trained scikit-learn pipeline once."""
    return joblib.load(model_path)


def show_metric(label, value):
    """Display one styled model metric."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-name">{label}</div>
            <div class="metric-number">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar
with st.sidebar:
    st.title("🛡️ MailGuard")
    st.caption("Machine learning email screening")
    st.divider()
    st.subheader("Model information")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Text features:** TF-IDF")
    st.write("**Categories:** Spam / Not Spam")
    st.divider()
    st.subheader("How it works")
    st.write("1. Enter or select an email.")
    st.write("2. Click Analyze email.")
    st.write("3. Review the prediction.")
    st.caption("Educational prototype. Do not rely on it as your only security filter.")


# Main heading
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ MailGuard</h1>
        <p>Spam Email Classifier · Check suspicious email text using machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Check required model file
if not MODEL_PATH.is_file():
    st.error("The trained model file was not found.")
    st.info("Run this command in the project terminal: python train_model.py")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("The model could not be loaded.")
    st.code(str(error))
    st.info("Try running: python train_model.py")
    st.stop()


# Model metrics
st.subheader("Model performance")

metrics = None
if METRICS_PATH.is_file():
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    except (OSError, json.JSONDecodeError):
        st.warning("The metrics file could not be read. Retrain the model to recreate it.")

if metrics:
    metric_columns = st.columns(4)
    metric_items = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1-score", metrics.get("f1_score")),
    ]

    for column, item in zip(metric_columns, metric_items):
        label, score = item
        display_value = f"{float(score):.1%}" if score is not None else "N/A"
        with column:
            show_metric(label, display_value)

    total_rows = metrics.get("dataset_rows", "N/A")
    test_rows = metrics.get("testing_rows", "N/A")
    st.caption(
        f"Evaluation used {test_rows} test emails from a sample dataset of {total_rows} emails."
    )
else:
    st.info("Model metrics are not available yet. Run python train_model.py.")

st.divider()

# Email input
st.subheader("Analyze an email")
st.write("Paste an email message below, or choose one of the sample messages.")

sample_messages = {
    "Choose a sample": "",
    "Example: suspicious prize message": (
        "Congratulations! You have won a free iPhone. "
        "Click here to claim your prize now."
    ),
    "Example: normal personal email": (
        "Hi, are we still meeting for lunch tomorrow? "
        "Please let me know what time works for you."
    ),
}

selected_sample = st.selectbox("Try a sample email", list(sample_messages.keys()))

email_text = st.text_area(
    "Email message",
    value=sample_messages[selected_sample],
    placeholder="Paste or type the email message here...",
    height=180,
)

if st.button("🔍 Analyze email", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter an email message first.")
    else:
        try:
            predicted_label = str(model.predict([email_text])[0]).lower()

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([email_text])[0]
                model_classes = [str(value).lower() for value in model.classes_]
                predicted_index = model_classes.index(predicted_label)
                confidence = float(probabilities[predicted_index])
            else:
                confidence = None

            st.subheader("Analysis result")

            if predicted_label == "spam":
                st.markdown(
                    """
                    <div class="spam-result">
                        <strong>🚨 Likely SPAM</strong><br>
                        The model classified this message as spam.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="ham-result">
                        <strong>✅ Likely NOT SPAM</strong><br>
                        The model classified this message as a legitimate email.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if confidence is not None:
                st.progress(confidence)
                st.caption(f"Model confidence: {confidence:.1%}")

            st.caption(
                "This is a machine-learning estimate, not a guarantee. "
                "Check links and requests carefully."
            )

        except Exception as error:
            st.error("Something went wrong while analyzing this message.")
            st.code(str(error))


# Project information
with st.expander("About this project"):
    st.markdown(
        """
        **Machine learning workflow**

        1. Email text is converted into numerical features using TF-IDF.
        2. TF-IDF includes individual words and two-word phrases.
        3. Multinomial Naive Bayes predicts the email category.
        4. The model is evaluated using accuracy, precision, recall and F1-score.

        **Dataset note:** This project uses a small, hand-prepared educational
        dataset. Its evaluation scores are for demonstration only and do not
        represent production-level spam detection.
        """
    )

# Confusion matrix
if metrics and isinstance(metrics.get("confusion_matrix"), list):
    matrix = metrics["confusion_matrix"]
    if len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2 for row in matrix):
        with st.expander("View evaluation details"):
            st.write(
                "Confusion matrix: rows show actual labels (ham, spam) "
                "and columns show predicted labels (ham, spam)."
            )
            st.dataframe(
                {
                    "Actual label": ["Ham", "Spam"],
                    "Predicted ham": [matrix[0][0], matrix[1][0]],
                    "Predicted spam": [matrix[0][1], matrix[1][1]],
                },
                hide_index=True,
                use_container_width=True,
            )

st.divider()
st.markdown(
    '<p class="footnote">MailGuard · Student machine learning project · '
    'Python | Scikit-learn | Streamlit</p>',
    unsafe_allow_html=True,
)
