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