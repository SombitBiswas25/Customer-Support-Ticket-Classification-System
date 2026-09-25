"""
Exploratory Data Analysis (EDA) Module.

This script performs detailed exploratory data analysis on the customer support
ticket dataset and exports high-resolution charts to the `screenshots/` directory:
1. Category distribution (Tickets per Category)
2. Priority distribution overall & stacked by category
3. Status distribution
4. Ticket text length distribution (words and characters)
5. Generates summary statistics and insights for documentation.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly or as a module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server/CLI execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.preprocessing import load_dataset, preprocess_dataframe, validate_and_summarize_dataset


def configure_plot_style():
    """Configures clean, modern plot aesthetic."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def plot_category_distribution(df: pd.DataFrame, output_path: str = "screenshots/category_distribution.png"):
    """
    Plots the count and proportion of tickets across categories.
    """
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("mako", n_colors=df["category"].nunique())
    
    order = df["category"].value_counts().index
    ax = sns.countplot(data=df, y="category", order=order, hue="category", palette=palette, legend=False)
    
    plt.title("Support Ticket Distribution by Category (N=200)")
    plt.xlabel("Number of Tickets")
    plt.ylabel("Category")
    
    # Annotate bar counts
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)} ({width/len(df)*100:.1f}%)",
                    (width + 0.5, p.get_y() + p.get_height() / 2.),
                    va="center", ha="left", fontsize=10, color="#333333", weight="bold")
        
    plt.xlim(0, max(df["category"].value_counts()) + 5)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved category distribution plot -> {output_path}")


def plot_priority_and_status(df: pd.DataFrame, output_path: str = "screenshots/priority_distribution.png"):
    """
    Plots a 2-panel chart showing Priority distribution and Priority-by-Category breakdown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Priority overall
    priority_order = ["Critical", "High", "Medium", "Low"]
    available_priorities = [p for p in priority_order if p in df["priority"].values]
    p_colors = {"Critical": "#e63946", "High": "#f4a261", "Medium": "#457b9d", "Low": "#2a9d8f"}
    palette = [p_colors.get(p, "#888888") for p in available_priorities]
    
    sns.countplot(data=df, x="priority", order=available_priorities, hue="priority", palette=palette, legend=False, ax=axes[0])
    axes[0].set_title("Tickets by Priority Level")
    axes[0].set_xlabel("Priority")
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{int(height)}",
                         (p.get_x() + p.get_width() / 2., height + 1),
                         ha="center", fontsize=10, weight="bold")
    axes[0].set_ylim(0, max(df["priority"].value_counts()) + 10)

    # Panel 2: Status overall
    status_order = df["status"].value_counts().index
    status_palette = sns.color_palette("viridis", n_colors=len(status_order))
    sns.countplot(data=df, x="status", order=status_order, hue="status", palette=status_palette, legend=False, ax=axes[1])
    axes[1].set_title("Tickets by Resolution Status")
    axes[1].set_xlabel("Status")
    axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f"{int(height)}",
                         (p.get_x() + p.get_width() / 2., height + 1),
                         ha="center", fontsize=10, weight="bold")
    axes[1].set_ylim(0, max(df["status"].value_counts()) + 10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved priority & status distribution plot -> {output_path}")


def plot_text_length_distribution(df: pd.DataFrame, output_path: str = "screenshots/text_length_distribution.png"):
    """
    Plots the distribution of ticket description length (words).
    """
    plt.figure(figsize=(9, 4.5))
    sns.histplot(df["word_count"], bins=15, kde=True, color="#3a86ff", edgecolor="black")
    plt.title("Distribution of Ticket Description Word Counts")
    plt.xlabel("Word Count per Ticket")
    plt.ylabel("Frequency")
    
    mean_words = df["word_count"].mean()
    median_words = df["word_count"].median()
    plt.axvline(mean_words, color="#d90429", linestyle="--", linewidth=1.5, label=f"Mean: {mean_words:.1f} words")
    plt.axvline(median_words, color="#06d6a0", linestyle=":", linewidth=2.0, label=f"Median: {median_words:.1f} words")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved text length distribution plot -> {output_path}")


def generate_eda_report(dataset_path: str = "data/customer_support_ticket_dataset_200.csv"):
    """
    Executes full EDA pipeline and prints comprehensive statistical summary.
    """
    os.makedirs("screenshots", exist_ok=True)
    configure_plot_style()

    print("\n" + "=" * 65)
    print("      EXPLORATORY DATA ANALYSIS (EDA) - SUPPORT TICKETS")
    print("=" * 65)

    df_raw = load_dataset(dataset_path)
    df = preprocess_dataframe(df_raw, drop_duplicates=False)

    total_tickets = len(df)
    categories = df["category"].nunique()
    print(f"Total Tickets Analyzed   : {total_tickets}")
    print(f"Total Unique Categories  : {categories}")
    print(f"Average Words per Ticket : {df['word_count'].mean():.2f}")
    print(f"Average Chars per Ticket : {df['char_length'].mean():.2f}")
    print(f"Min / Max Words          : {df['word_count'].min()} / {df['word_count'].max()}")

    print("\n--- Category Breakdown ---")
    cat_df = df["category"].value_counts().reset_index()
    cat_df.columns = ["Category", "Count"]
    cat_df["Percentage"] = (cat_df["Count"] / total_tickets * 100).round(1).astype(str) + "%"
    print(cat_df.to_string(index=False))

    print("\n--- Priority Breakdown ---")
    prio_df = df["priority"].value_counts().reset_index()
    prio_df.columns = ["Priority", "Count"]
    prio_df["Percentage"] = (prio_df["Count"] / total_tickets * 100).round(1).astype(str) + "%"
    print(prio_df.to_string(index=False))

    print("\n--- Status Breakdown ---")
    stat_df = df["status"].value_counts().reset_index()
    stat_df.columns = ["Status", "Count"]
    stat_df["Percentage"] = (stat_df["Count"] / total_tickets * 100).round(1).astype(str) + "%"
    print(stat_df.to_string(index=False))

    # Generate visual charts
    plot_category_distribution(df)
    plot_priority_and_status(df)
    plot_text_length_distribution(df)

    print("\n[EDA] All charts generated successfully in screenshots/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys
    path = "data/customer_support_ticket_dataset_200.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif not os.path.exists(path):
        path = "customer_support_ticket_dataset_200.csv"
    generate_eda_report(path)
