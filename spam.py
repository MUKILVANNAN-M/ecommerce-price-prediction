import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def classify_review(review):
    response = model.generate_content(f"Classify as Spam or Not Spam: {review}")
    return response.text.strip()

def review_summarization(review):
    response = model.generate_content(f"Summarize this review in 100 words: {review}")
    return response.text.strip()

def quality_pred(review):
    response = model.generate_content(f"Classify quality as Good, Average or Bad: {review}")
    return response.text.strip()

def topic_model(review):
    response = model.generate_content(f"Give topic analysis of this review: {review}")
    return response.text.strip()
