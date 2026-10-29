import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_average_roi_by_country(df):
    if "ROI" not in df.columns:
        df["ROI"] = np.where(
            df["budget"].ne(0),
            (df["revenue"] - df["budget"]) / df["budget"],
            np.nan,
        )

    country_roi = (
        df.assign(prod_country=df["production_countries"].fillna("").str.split(","))
        .explode("prod_country")
        .assign(prod_country=lambda x: x["prod_country"].str.strip())
        .loc[lambda x: x["prod_country"].ne("")]
        [["prod_country", "ROI"]]
    )

    avg_roi_by_country = (
        country_roi.groupby("prod_country")["ROI"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )

    top_countries = avg_roi_by_country.head(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    top_countries.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Average ROI by Production Country")
    ax.set_xlabel("Production Country")
    ax.set_ylabel("Average ROI")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

def plot_genre_distribution(df):
    genre_series = (
        df["genres"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
    )

    genre_counts = genre_series.value_counts()
    genre_pct = genre_counts / genre_counts.sum() * 100

    plt.figure(figsize=(10, 8))
    plt.pie(
        genre_pct.head(10).values,
        labels=genre_pct.head(10).index,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.6,
    )
    plt.title("Percentage of Movies by Genre")
    plt.axis("equal")
    plt.show()

def plot_runtime_vs_revenue(df):
    sub = df[['runtime_minutes', 'revenue']].copy().dropna()
    sub = sub[(sub['runtime_minutes'] > 0) & (sub['revenue'] > 0)]

    x = sub['runtime_minutes']
    y = np.log1p(sub['revenue'])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.25, s=12)

    res = np.polyfit(x, y, 1)
    y_fit = np.polyval(res, x)
    plt.plot(x, y_fit, color='red', linewidth=2, label='Regression line')

    plt.title('Runtime Minutes vs Log-Revenue Regression')
    plt.xlabel('Runtime Minutes')
    plt.ylabel('log(1 + revenue)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_franchise_vs_standalone_revenue(df):
    """Boxplot comparing log-revenue for franchise vs. standalone movies."""
    sub = df[['revenue', 'belongs_to_collection']].copy().dropna(subset=['revenue'])
    sub = sub[sub['revenue'] > 0].copy()
    sub['log_revenue'] = np.log1p(sub['revenue'])
    sub['group'] = np.where(sub['belongs_to_collection'].notna(), 'Franchise', 'Standalone')

    groups = [sub.loc[sub['group'] == g, 'log_revenue'] for g in ['Franchise', 'Standalone']]

    plt.figure(figsize=(6, 6))
    plt.boxplot(groups, tick_labels=['Franchise', 'Standalone'], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
    plt.title('Revenue: Franchise vs. Standalone Movies')
    plt.ylabel('log(1 + revenue)')
    plt.tight_layout()
    plt.show()

def plot_language_revenue(df):
    """Boxplot comparing log-revenue for English vs. other original languages."""
    sub = df[['revenue', 'original_language']].copy().dropna()
    sub = sub[sub['revenue'] > 0].copy()
    sub['log_revenue'] = np.log1p(sub['revenue'])
    sub['group'] = np.where(sub['original_language'] == 'en', 'English', 'Other')

    groups = [sub.loc[sub['group'] == g, 'log_revenue'] for g in ['English', 'Other']]

    plt.figure(figsize=(6, 6))
    plt.boxplot(groups, tick_labels=['English', 'Other'], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
    plt.title('Revenue: English vs. Other Original Language')
    plt.ylabel('log(1 + revenue)')
    plt.tight_layout()
    plt.show()

def plot_rating_vs_revenue(df, min_vote_count=20):
    """Scatter + regression line for audience rating (vote_average) vs. log-revenue."""
    sub = df[['vote_average', 'revenue', 'vote_count']].copy().dropna()
    sub = sub[(sub['revenue'] > 0) & (sub['vote_count'] >= min_vote_count)]

    x = sub['vote_average']
    y = np.log1p(sub['revenue'])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.25, s=12)

    res = np.polyfit(x, y, 1)
    y_fit = np.polyval(res, x)
    order = np.argsort(x)
    plt.plot(x.values[order], y_fit[order], color='red', linewidth=2, label='Regression line')

    plt.title('Audience Rating vs Log-Revenue')
    plt.xlabel('Vote Average')
    plt.ylabel('log(1 + revenue)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_genre_revenue_boxplot(df, top_n=6):
    """Boxplot of log-revenue across the top primary genres, ordered by median revenue."""
    sub = df[['genres', 'revenue']].copy().dropna()
    sub = sub[sub['revenue'] > 0].copy()
    sub['log_revenue'] = np.log1p(sub['revenue'])
    sub['primary_genre'] = sub['genres'].str.split(',').str[0].str.strip()

    top_genres = sub['primary_genre'].value_counts().head(top_n).index
    sub = sub[sub['primary_genre'].isin(top_genres)]

    order = sub.groupby('primary_genre')['log_revenue'].median().sort_values(ascending=False).index
    groups = [sub.loc[sub['primary_genre'] == g, 'log_revenue'] for g in order]

    plt.figure(figsize=(10, 6))
    plt.boxplot(groups, tick_labels=list(order), patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
    plt.title('Revenue by Primary Genre')
    plt.ylabel('log(1 + revenue)')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()