import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv

load_dotenv()

class SentimentResult(BaseModel):
    sentiment: Literal['positive', 'neutral', 'negative'] = Field(
        description="The sentiment of the text"
    )
    confidence_score: float = Field(
        description="Confidence score of the sentiment extraction between 0.0 and 1.0"
    )
    key_topics: List[str] = Field(
        description="List of key topics extracted from the text"
    )
    action_required: bool = Field(
        description="Whether the text requires follow-up action"
    )

MOCK_DATASET = [
    "I absolutely love the new dashboard features. It makes my workflow so much faster!",
    "The app keeps crashing when I try to upload a PDF file. Please fix this immediately.",
    "Can you add a dark mode? It's really hard to use this at night.",
    "Customer service was okay, but the wait time was a bit too long.",
    "This is the worst experience I've ever had. I want a refund right now."
]

def simulate_llm_extraction(text: str) -> dict:
    #API latency
    latency = random.uniform(0.5, 2.5)
    time.sleep(latency)
    
    input_tokens = len(text.split()) * 1.5
    output_tokens = random.uniform(20, 50)
    total_tokens = int(input_tokens + output_tokens)
    
    text_lower = text.lower()
    
    if "love" in text_lower or "great" in text_lower:
        sentiment = "positive"
        action_required = False
        key_topics = ["dashboard", "workflow", "usability"]
    elif "worst" in text_lower or "crashing" in text_lower or "refund" in text_lower:
        sentiment = "negative"
        action_required = True
        key_topics = ["crash", "pdf_upload", "refund", "stability"]
    elif "add" in text_lower or "request" in text_lower or "dark mode" in text_lower:
        sentiment = "neutral"
        action_required = False
        key_topics = ["feature_request", "dark_mode"]
    else:
        sentiment = "neutral"
        action_required = bool(random.choice([True, False]))
        key_topics = ["general", "customer_service"]
        
    result = SentimentResult(
        sentiment=sentiment,
        confidence_score=round(random.uniform(0.7, 0.99), 2),
        key_topics=key_topics,
        action_required=action_required
    )
    
    return {
        "text": text,
        "sentiment": result.sentiment,
        "confidence_score": result.confidence_score,
        "key_topics": ", ".join(result.key_topics),
        "action_required": result.action_required,
        "latency_sec": round(latency, 3),
        "total_tokens": total_tokens
    }

def process_dataset():
    results = []
    print("Starting extraction pipeline...")
    for i, text in enumerate(MOCK_DATASET):
        print(f"Processing text {i+1}/{len(MOCK_DATASET)}...")
        res = simulate_llm_extraction(text)
        results.append(res)
    
    df = pd.DataFrame(results)
    
    print("\n--- Extraction Results ---")
    print(df.to_string())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    plot_results(df)
    
def plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sentiment_counts = df['sentiment'].value_counts()
    color_map = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    colors = [color_map.get(s, 'blue') for s in sentiment_counts.index]
    
    axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    axes[0].set_title('Sentiment Distribution', fontsize=14)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xlabel('Sentiment Type', fontsize=12)
    
    axes[1].bar(df.index + 1, df['latency_sec'], color='skyblue', edgecolor='black')
    axes[1].set_title('Extraction Latency (sec) per Document', fontsize=14)
    axes[1].set_ylabel('Latency (Seconds)', fontsize=12)
    axes[1].set_xlabel('Document ID', fontsize=12)
    axes[1].set_xticks(df.index + 1)
    
    plt.tight_layout()
    plot_path = 'summary_plot.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved successfully as '{plot_path}'.")

if __name__ == "__main__":
    process_dataset()
