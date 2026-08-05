import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def calculate_roi(df):
    """Calculate the Return on Investment (ROI) for each movie."""
    df["ROI"] = (df["revenue"] - df["budget"]) / df["budget"]
    return df[["revenue", "budget", "ROI"]]

def run_regression_analysis(df):
    """Run regression analysis to determine the relationship between budget and revenue."""
    sub = df[['budget', 'revenue', 'belongs_to_collection']].copy().dropna()
    sub = sub[(sub['budget'] > 0) & (sub['revenue'] > 0)].copy()

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