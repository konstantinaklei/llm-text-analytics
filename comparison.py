import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
from typing import List, Literal

class SentimentResult(BaseModel):
    sentiment: Literal['positive', 'neutral', 'negative']
    confidence_score: float
    key_topics: List[str]
    action_required: bool

DATASET = [
    "I absolutely love the new dashboard features. It makes my workflow so much faster!",
    "The app keeps crashing when I try to upload a PDF file. Please fix this immediately.",
    "Can you add a dark mode? It's really hard to use this at night.",
    "Customer service was okay, but the wait time was a bit too long.",
    "This is the worst experience I've ever had. I want a refund right now."
]

def simulate_llm_call(text: str, strategy: str) -> dict:
    """
    Simulates an LLM API call varying by strategy:
    - 'strict_json': Lower input tokens, lower/standard latency, moderate consistency.
    - 'few_shot': Higher input tokens (due to examples), slightly higher latency, high consistency.
    """
    base_input_tokens = len(text.split()) * 1.5
    
    if strategy == "strict_json":
        # Strict JSON prompt instructions without examples
        input_tokens = base_input_tokens + 50
        latency = random.uniform(0.6, 1.5)
        consistency_score = random.uniform(0.75, 0.90)
    elif strategy == "few_shot":
        # Prompt includes multiple rich examples
        input_tokens = base_input_tokens + 250
        latency = random.uniform(1.2, 2.8)
        consistency_score = random.uniform(0.90, 0.99)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
        
    time.sleep(latency / 5) # Speed up simulation
    output_tokens = random.uniform(20, 45)
    total_tokens = int(input_tokens + output_tokens)
    
    success = random.random() < consistency_score
    
    return {
        "text_id": DATASET.index(text) + 1,
        "strategy": strategy,
        "latency_sec": round(latency, 3),
        "total_tokens": total_tokens,
        "success": success,
        "consistency_score": round(consistency_score, 2)
    }

def run_comparison():
    strategies = ["strict_json", "few_shot"]
    results = []
    
    print("Running Prompt Strategy Comparison...")
    for text in DATASET:
        for strategy in strategies:
            res = simulate_llm_call(text, strategy)
            results.append(res)
            
    df = pd.DataFrame(results)
    
    print("\n--- Comparative DataFrame ---")
    print(df.to_string())
    
    summary = df.groupby('strategy').agg(
        avg_latency=('latency_sec', 'mean'),
        avg_tokens=('total_tokens', 'mean'),
        success_rate=('success', 'mean'),
        avg_consistency=('consistency_score', 'mean')
    ).reset_index()
    
    print("\n--- Summary Table ---")
    print(summary.to_string(index=False))

    plot_comparison(df, summary)

def plot_comparison(df: pd.DataFrame, summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = ['#FFA07A', '#20B2AA']
    
    df.boxplot(column='latency_sec', by='strategy', ax=axes[0], patch_artist=True)
    axes[0].set_title('Latency Comparison')
    axes[0].set_ylabel('Latency (Seconds)')
    axes[0].set_xlabel('Strategy')
    
    axes[1].bar(summary['strategy'], summary['avg_tokens'], color=colors)
    axes[1].set_title('Average Total Tokens Used')
    axes[1].set_ylabel('Tokens')
    axes[1].set_xlabel('Strategy')
    
    success_pct = summary['success_rate'] * 100
    axes[2].bar(summary['strategy'], success_pct, color=colors)
    axes[2].set_title('Extraction Success Rate (%)')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_xlabel('Strategy')
    axes[2].set_ylim(0, 100)
    
    fig.suptitle("Prompt Strategy Comparison: Strict JSON vs Few-Shot", fontsize=16)
    
    plt.tight_layout()
    plot_path = "strategy_comparison_plot.png"
    plt.savefig(plot_path)
    print(f"\nComparative plot saved successfully as '{plot_path}'.")

if __name__ == "__main__":
    run_comparison()
