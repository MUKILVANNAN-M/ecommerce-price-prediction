from pyfin_sentiment.model import SentimentModel

SentimentModel.download("small")
model = SentimentModel("small")

def financial_sentiment_analysis(text):
    sentiment = model.predict([text])
    return {"predicted_sentiment": sentiment[0]}
