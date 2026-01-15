import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

def set_korean_font():
    # Detect OS and set font
    if os.name == 'nt': # Windows
        plt.rc('font', family='Malgun Gothic')
    elif os.uname().sysname == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    else: # Linux
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)

def main():
    set_korean_font()
    
    # Load Data
    csv_path = "result/youtube_pairs_detail.csv"
    if not os.path.exists(csv_path):
        print("Data file not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Create Output Directory
    os.makedirs("result/plots", exist_ok=True)
    
    # 1. Sentiment Comparison (Stacked Bar)
    # Average ratios per group
    sentiment_cols = ['positive_ratio', 'negative_ratio', 'neutral_ratio']
    avg_sentiment = df.groupby('group_name')[sentiment_cols].mean()
    
    ax = avg_sentiment.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', alpha=0.8)
    plt.title('Average Sentiment Ratio by Group')
    plt.ylabel('Ratio')
    plt.xlabel('Group')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('result/plots/sentiment_stacked_bar.png')
    plt.close()
    
    # 2. Key Metrics Distribution (Box Plot)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.boxplot(x='group_name', y='view_count', data=df, ax=axes[0], palette='Set2')
    axes[0].set_title('View Count Distribution')
    axes[0].set_yscale('log') # Log scale fo views
    
    sns.boxplot(x='group_name', y='like_count', data=df, ax=axes[1], palette='Set2')
    axes[1].set_title('Like Count Distribution')
    axes[1].set_yscale('log')
    
    sns.boxplot(x='group_name', y='comment_count', data=df, ax=axes[2], palette='Set2')
    axes[2].set_title('Comment Count Distribution')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('result/plots/metrics_boxplot.png')
    plt.close()
    
    # 3. Positive Ratio Distribution (KDE / Hist)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='positive_ratio', hue='group_name', kde=True, bins=10, palette='Set1', alpha=0.6)
    plt.title('Distribution of Positive Sentiment Ratio')
    plt.xlabel('Positive Ratio (0.0 - 1.0)')
    plt.ylabel('Count')
    plt.savefig('result/plots/positive_ratio_dist.png')
    plt.close()
    
    # 4. Correlation Heatmap
    # Filter numeric cols
    numeric_df = df[['view_count', 'like_count', 'comment_count', 'positive_ratio', 'negative_ratio']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('result/plots/correlation_heatmap.png')
    plt.close()
    
    print("Optimization Complete. Plots saved to result/plots/")

if __name__ == "__main__":
    main()
