import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background-color: transparent;
}

.block-container{
    padding-top: 2rem;
}

.section-header{
    font-size:24px;
    font-weight:600;
    margin-top:15px;
    margin-bottom:10px;
}

[data-testid="metric-container"]{
    border-radius:12px;
    padding:15px;
}

.stButton > button{
    width:100%;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

textarea{
    border-radius:10px !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================

rnn_model = tf.keras.models.load_model(
    "models/rnn_model.h5"
)

lstm_model = tf.keras.models.load_model(
    "models/lstm_model.h5"
)

gru_model = tf.keras.models.load_model(
    "models/gru_model.h5"
)

# =====================================================
# LOAD TOKENIZER
# =====================================================

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# =====================================================
# HEADER
# =====================================================

# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<h1 style='text-align:center;'>
🎬 Movie Review Sentiment Analysis System
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<h3 style='text-align:center; color:gray;'>
Deep Learning Based Sentiment Classification using
SimpleRNN • LSTM • GRU
</h3>
""",
unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Settings")

selected_model = st.sidebar.radio(
    "Select Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_sentiment(review, model):

    sequence = tokenizer.texts_to_sequences(
        [review]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    probability = model.predict(
        padded,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability
        if probability >= 0.5
        else (1 - probability)
    )

    return sentiment, confidence, probability


# =====================================================
# REVIEW INPUT
# =====================================================

review = st.text_area(
    "Enter your movie review here...",
    height=180,
    placeholder="Example: This movie was absolutely fantastic and I enjoyed every minute."
)

# =====================================================
# ANALYZE BUTTON
# =====================================================

if st.button("🔍 Analyze Review"):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        model_dict = {
            "SimpleRNN": rnn_model,
            "LSTM": lstm_model,
            "GRU": gru_model
        }

        sentiment, confidence, probability = predict_sentiment(
            review,
            model_dict[selected_model]
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"Sentiment : {sentiment}"
            )

            st.info(
                f"Confidence : {confidence * 100:.2f}%"
            )

        with col2:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={
                        'text': "Confidence Score"
                    },
                    gauge={
                        'axis': {
                            'range': [0, 100]
                        }
                    }
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        # ==========================================
        # PROBABILITY CALCULATIONS
        # ==========================================

        positive = probability * 100
        negative = (1 - probability) * 100

        st.subheader(
            "Prediction Visualization"
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            pie = px.pie(
                names=[
                    "Positive",
                    "Negative"
                ],
                values=[
                    positive,
                    negative
                ],
                hole=0.5,
                title="Probability Distribution"
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )
            with chart_col2:

                bar = px.bar(
                    x=["Positive", "Negative"],
                    y=[positive, negative],
                    title="Confidence Comparison"
            )

                st.plotly_chart(
                    bar,
                    use_container_width=True
                )
            st.markdown("---")
            st.subheader("📊 Compare All Models")

            rnn_sent, rnn_conf, _ = predict_sentiment(
                review,
                rnn_model
            )

            lstm_sent, lstm_conf, _ = predict_sentiment(
                review,
                lstm_model
        )

            gru_sent, gru_conf, _ = predict_sentiment(
                review,
                gru_model
        )

            compare_df = pd.DataFrame({
            "Model": [
                "SimpleRNN",
                "LSTM",
                "GRU"
            ],
            "Prediction": [
                rnn_sent,
                lstm_sent,
                gru_sent
            ],
            "Confidence (%)": [
                round(rnn_conf * 100, 2),
                round(lstm_conf * 100, 2),
                round(gru_conf * 100, 2)
            ]
        })

            st.dataframe(
            compare_df,
            use_container_width=True
        )

        # ==========================================
        # BEST MODEL
        # ==========================================

            scores = {
            "SimpleRNN": rnn_conf,
            "LSTM": lstm_conf,
            "GRU": gru_conf
        }

            best_model = max(
            scores,
            key=scores.get
        )

            st.success(
            f"🏆 Best Confidence Model: {best_model}"
        )

        # ==========================================
        # REVIEW STATISTICS
        # ==========================================

            st.markdown("---")
            st.subheader("📈 Review Statistics")

            words = len(review.split())
            chars = len(review)

            stat1, stat2, stat3 = st.columns(3)

            stat1.metric(
            "Words",
            words
        )

            stat2.metric(
            "Characters",
            chars
        )

            if words < 20:
                review_type = "Short"

            elif words < 50:
                review_type = "Medium"

            else:
                review_type = "Long"

            stat3.metric(
            "Review Length",
            review_type
        )

        # ==========================================
        # DOWNLOAD RESULT
        # ==========================================

            st.markdown("---")

            result_df = pd.DataFrame({
            "Review": [review],
            "Prediction": [sentiment],
            "Confidence": [
                round(confidence * 100, 2)
            ]
        })

            csv = result_df.to_csv(
            index=False
        )

            st.download_button(
            label="⬇ Download Result",
            data=csv,
            file_name="prediction.csv",
            mime="text/csv"
        )
        # ==========================================
        # COMPARE ALL MODELS
        # ===============