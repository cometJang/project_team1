
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

def set_korean_font():
    if os.name == 'nt': 
        plt.rc('font', family='Malgun Gothic')
    elif os.uname().sysname == 'Darwin': 
        plt.rc('font', family='AppleGothic')
    else: 
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)

def main():
    set_korean_font()
    
    # Load Data (Solar Result)
    csv_path = "result/youtube_pairs_detail_solar.csv"
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    os.makedirs("result/plots_solar", exist_ok=True)
    
    # 1. Sentiment Comparison
    sentiment_cols = ['positive_ratio', 'negative_ratio', 'neutral_ratio']
    if all(c in df.columns for c in sentiment_cols):
        avg_sentiment = df.groupby('group_name')[sentiment_cols].mean()
        
        ax = avg_sentiment.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', alpha=0.8)
        plt.title('Average Sentiment Ratio by Group (Solar)')
        plt.ylabel('Ratio')
        plt.xlabel('Group')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('result/plots_solar/sentiment_stacked_bar.png')
        plt.close()
    
    # 2. Key Metrics
    if not df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sns.boxplot(x='group_name', y='view_count', data=df, ax=axes[0], palette='Set2')
        axes[0].set_title('View Count')
        axes[0].set_yscale('log')
        
        sns.boxplot(x='group_name', y='like_count', data=df, ax=axes[1], palette='Set2')
        axes[1].set_title('Like Count')
        axes[1].set_yscale('log')
        
        sns.boxplot(x='group_name', y='comment_count', data=df, ax=axes[2], palette='Set2')
        axes[2].set_title('Comment Count')
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('result/plots_solar/metrics_boxplot.png')
        plt.close()
    
    # 3. Positive Ratio Dist
    if 'positive_ratio' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='positive_ratio', hue='group_name', kde=True, bins=10, palette='Set1', alpha=0.6)
        plt.title('Distribution of Positive Sentiment Ratio (Solar)')
        plt.savefig('result/plots_solar/positive_ratio_dist.png')
        plt.close()
    
    print("Plots saved to result/plots_solar/")

if __name__ == "__main__":
    main()
