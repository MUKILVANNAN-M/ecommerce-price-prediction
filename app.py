import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

from spam import quality_pred, topic_model, classify_review, review_summarization
from sentiment import financial_sentiment_analysis

# Load model and dataset
model = joblib.load('random_forest_model.pkl')
data = pd.read_csv('Data.csv')

# Clean price columns
def clean_price(price):
    if isinstance(price, str):
        return float(price.replace("₹", "").replace(",", "").strip())
    return price

data['actual_price'] = data['actual_price'].apply(clean_price)
data['discount_percentage'] = data['discount_percentage'].apply(lambda x: float(str(x).replace("%", "")))

# Streamlit UI
st.title("📊 Amazon Product Price Prediction & Review Intelligence System")

product_url = st.text_input("Enter Product URL:")

if product_url:
    product_data = data[data['product_link'] == product_url]

    if not product_data.empty:
        product = product_data.iloc[0]

        st.subheader("📦 Product Details")
        st.write(f"**Name:** {product['product_name']}")
        st.write(f"**Category:** {product['category']}")
        st.write(f"**Price:** ₹{product['actual_price']}")
        st.write(f"**Discount:** {product['discount_percentage']}%")
        st.write(f"**Rating:** {product['rating']}")
        st.write(f"**Reviews:** {product['rating_count']}")
        st.image(product['img_link'], use_container_width=True)

        # Feature Engineering
        discount_amount = product['actual_price'] * (product['discount_percentage'] / 100)
        popularity_score = product['rating'] * product['rating_count']

        features = np.array([[ 
            product['actual_price'],
            product['discount_percentage'],
            product['rating'],
            product['rating_count'],
            discount_amount,
            popularity_score,
            0  # category placeholder
        ]])

        if st.button("🔮 Predict Price"):
            predicted = model.predict(features)
            st.success(f"Predicted Price: ₹{predicted[0]:.2f}")

            review = product['review_content']

            # Sentiment
            sentiment = financial_sentiment_analysis(review)
            st.subheader("📈 Sentiment Analysis")
            st.info(f"Sentiment: {sentiment['predicted_sentiment']}")

            # Spam
            st.subheader("🚫 Spam Detection")
            st.info(classify_review(review))

            # Summary
            st.subheader("📝 Summary")
            st.info(review_summarization(review))

            # Quality
            st.subheader("⭐ Product Quality")
            st.info(quality_pred(review))

            # Topic
            st.subheader("📊 Topic Analysis")
            st.info(topic_model(review))

    else:
        st.error("Product not found ❌")
