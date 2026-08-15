import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def calculate_roi(df):
    """Calculate the Return on Investment (ROI) for each movie."""
    df["ROI"] = (df["revenue"] - df["budget"]) / df["budget"]
    return df[["revenue", "budget", "ROI"]]

def run_regression_analysis(df):
    """Run regression analysis to determine the relationship between budget and revenue."""
    sub = df[['budget', 'revenue', 'belongs_to_collection']].copy()
    sub = sub.dropna(subset=['budget', 'revenue'])
    sub = sub[(sub['budget'] > 0) & (sub['revenue'] > 0)].copy()

    # belongs_to_collection is NaN for standalone movies, not "missing" -- keep those
    # rows and treat NaN as franchise=0 rather than dropping them.
    sub['franchise'] = sub['belongs_to_collection'].notna().astype(int)
    sub['log_budget'] = np.log1p(sub['budget'])
    sub['log_revenue'] = np.log1p(sub['revenue'])

    X = sm.add_constant(sub[['log_budget', 'franchise']])
    y = sub['log_revenue']
    model = sm.OLS(y, X).fit()

    return {
        'rows_used': len(sub),
        'budget_coef': model.params['log_budget'],
        'franchise_coef': model.params['franchise'],
        'franchise_p_value': model.pvalues['franchise'],
        'r_squared': model.rsquared,
        'summary': model.summary()
    }

def hypothesis_test_runtime_revenue(df):
    """Perform hypothesis testing for the relationship between runtime and revenue."""
    sub = df[['runtime_minutes', 'revenue']].copy().dropna()
    sub = sub[(sub['runtime_minutes'] > 0) & (sub['revenue'] > 0)]

    x = sub['runtime_minutes']
    y = np.log1p(sub['revenue'])

    res = stats.linregress(x, y)
    return {
        'rows_used': len(sub),
        'slope': res.slope,
        'intercept': res.intercept,
        'r': res.rvalue,
        'p_value': res.pvalue,
        'stderr': res.stderr
    }

def hypothesis_test_franchise_revenue(df):
    """Welch's t-test: do franchise movies earn more than standalone ones?"""
    sub = df[['revenue', 'belongs_to_collection']].copy()
    sub = sub.dropna(subset=['revenue'])
    sub = sub[sub['revenue'] > 0].copy()
    sub['log_revenue'] = np.log1p(sub['revenue'])

    franchise = sub.loc[sub['belongs_to_collection'].notna(), 'log_revenue']
    standalone = sub.loc[sub['belongs_to_collection'].isna(), 'log_revenue']
    t_stat, p_value = stats.ttest_ind(franchise, standalone, equal_var=False)

    return {
        'franchise_n': len(franchise),
        'standalone_n': len(standalone),
        'franchise_mean_log_revenue': franchise.mean(),
        'standalone_mean_log_revenue': standalone.mean(),
        't_stat': t_stat,
        'p_value': p_value,
    }

def hypothesis_test_language_revenue(df):
    """Welch's t-test: does original language (English vs. other) relate to revenue?"""
    sub = df[['revenue', 'original_language']].copy().dropna()
    sub = sub[sub['revenue'] > 0].copy()
    sub['log_revenue'] = np.log1p(sub['revenue'])

    english = sub.loc[sub['original_language'] == 'en', 'log_revenue']
    other = sub.loc[sub['original_language'] != 'en', 'log_revenue']
    t_stat, p_value = stats.ttest_ind(english, other, equal_var=False)

    return {
        'english_n': len(english),
        'other_n': len(other),
        'english_mean_log_revenue': english.mean(),
        'other_mean_log_revenue': other.mean(),
        't_stat': t_stat,
        'p_value': p_value,
    }

def hypothesis_test_rating_revenue(df, min_vote_count=20):
    """Pearson correlation: does audience rating (vote_average) relate to revenue?

    Movies with very few votes are dropped since their average rating is unreliable.
    """
    sub = df[['vote_average', 'revenue', 'vote_count']].copy().dropna()
    sub = sub[(sub['revenue'] > 0) & (sub['vote_count'] >= min_vote_count)]
    log_revenue = np.log1p(sub['revenue'])

    r, p_value = stats.pearsonr(sub['vote_average'], log_revenue)

    return {
        'rows_used': len(sub),
        'r': r,
        'p_value': p_value,
    }

def hypothesis_test_genre_revenue(df, top_n=6):
    """One-way ANOVA + Tukey HSD: does mean revenue differ across the top genres?

    Each movie is assigned to its first listed genre so groups are non-overlapping.
    """
    sub = df[['genres', 'revenue']].copy().dropna()
    sub = sub[sub['revenue'] > 0].copy()
    sub['log_revenue'] = np.log1p(sub['revenue'])
    sub['primary_genre'] = sub['genres'].str.split(',').str[0].str.strip()

    top_genres = sub['primary_genre'].value_counts().head(top_n).index
    sub = sub[sub['primary_genre'].isin(top_genres)]

    groups = [g['log_revenue'].values for _, g in sub.groupby('primary_genre')]
    f_stat, p_value = stats.f_oneway(*groups)
    tukey = pairwise_tukeyhsd(sub['log_revenue'], sub['primary_genre'])

    return {
        'rows_used': len(sub),
        'genres_compared': list(top_genres),
        'f_stat': f_stat,
        'p_value': p_value,
        'tukey_summary': tukey.summary(),
    }