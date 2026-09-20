import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
client = genai.Client()

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

from google.genai.errors import ServerError, ClientError

def extract_with_gemini(text: str) -> dict:
    """ Google Gemini API to extract structured sentiment."""
    for attempt in range(5):
        try:
            start_time = time.perf_counter()
            
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=text,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SentimentResult,
                ),
            )
            
            latency = time.perf_counter() - start_time
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            
            result = SentimentResult.model_validate_json(response.text)
            break
        except (ServerError, ClientError) as e:
            if attempt < 4:
                print(f"API Error ({e}). Retrying in 5 seconds (attempt {attempt+1}/5)...")
                time.sleep(5)
            else:
                raise e
    
    return {
        "text": text,
        "sentiment": result.sentiment,
        "confidence_score": result.confidence_score,
        "key_topics": ", ".join(result.key_topics),
        "action_required": result.action_required,
        "latency_sec": round(latency, 3),
        "total_tokens": prompt_tokens + completion_tokens
    }

def process_dataset():
    results = []
    print("Starting extraction pipeline...")
    for i, text in enumerate(MOCK_DATASET):
        print(f"Processing text {i+1}/{len(MOCK_DATASET)}...")
        res = extract_with_gemini(text)
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
