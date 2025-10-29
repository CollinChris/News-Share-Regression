"""
missingness_test.py: A module to test missing data mechanisms (MCAR, MAR, MNAR) in a given DataFrame.

Usage:
- Import and call check_missingness(df) to analyze your dataset.
- Run the script directly for a demo with simulated data.

Requirements: pandas, numpy, scipy, matplotlib, seaborn.
"""

"""
import sys
import os

# Simplified: Assume notebook is in project root (CWD), add root and src to path
project_root = os.getcwd()  # CWD is project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))

# Verify paths were added
print("Updated sys.path includes:")
for path in sys.path[:3]:  # Show first few for confirmation
    print(f"  - {path}")

# Now import the module
from missingness import check_missingness

# Load your data and test
df = pd.read_csv('data/student_data.csv')  # Adjust path as needed
results = check_missingness(df)
print(results)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def check_missingness(df, alpha=0.05):
    """
    Check missing data mechanisms (MCAR, MAR, MNAR) in the input DataFrame.
    
    Parameters:
    - df: Input DataFrame to analyze.
    - alpha: Significance level for chi-square test (default 0.05).
    
    Returns:
    - dict: Results including missing rates, test statistics, p-values, and decisions for each column.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    results = {}
    missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    
    # Debug: Print detected missing and complete columns
    print("Detected missing columns:", missing_cols)
    complete_cols = []
    for col in df.columns:
        if df[col].isnull().sum() == 0 and col not in missing_cols:
            # Convert to numeric to handle mixed dtypes
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > 0:  # Only include if some numeric values
                df[col + '_numeric'] = numeric_col  # Temp column for binning
                complete_cols.append(col + '_numeric')
            else:
                print(f"Warning: Skipping column '{col}' - could not convert to numeric (all non-numeric).")
    print("Detected complete columns (numeric):", [c.replace('_numeric', '') for c in complete_cols])
    
    if not missing_cols:
        print("No missing values found in the DataFrame.")
        return {'overall': 'No missingness detected'}
    
    for missing_col in missing_cols:
        missing_rate = df[missing_col].isnull().mean()
        missing_indicator = df[missing_col].isnull().astype(int)
        
        chi2_stats = []
        p_values = []
        decisions = []
        
        for other_col in complete_cols:
            original_col = other_col.replace('_numeric', '')  # Map back for logging
            try:
                # Bin the numeric version
                binned = pd.cut(df[other_col], bins=min(5, len(df[other_col].dropna().unique())), duplicates='drop')
                contingency = pd.crosstab(missing_indicator, binned)
                if contingency.shape[1] < 2 or contingency.shape[0] < 2:
                    p_values.append(np.nan)
                    decisions.append('Skip: Insufficient categories')
                    continue
                
                chi2, p = stats.chi2_contingency(contingency)[0:2]
                chi2_stats.append(chi2)
                p_values.append(p)
                decisions.append('Independent (MCAR)' if p > alpha else 'Dependent (MAR)')
            except Exception as e:
                print(f"Warning: Skipping test for {missing_col} vs. {original_col}: {str(e)}")
                p_values.append(np.nan)
                decisions.append('Skip: Binning failed')
                continue
        
        all_independent = all(p > alpha for p in p_values if not np.isnan(p))
        overall_decision = 'MCAR' if all_independent else 'Likely MAR'
        results[missing_col] = {
            'missing_rate': missing_rate,
            'chi2_stats': chi2_stats,
            'p_values': p_values,
            'decisions': decisions,
            'overall_decision': overall_decision,
            'note': 'MNAR cannot be ruled out without domain knowledge; check distributions manually.'
        }
    
    # Clean up temp columns
    for col in list(df.columns):
        if col.endswith('_numeric'):
            df.drop(col, axis=1, inplace=True)
    
    # Visualize missingness
    visualize_missingness(df)
    
    # Print results in a readable format (full version - no limit)
    print("\nMissingness Analysis Results:")
    for col, res in results.items():
        print(f"\nColumn: {col}")
        print(f"  Missing Rate: {res['missing_rate']:.3f}")
        print(f"  Overall Decision: {res['overall_decision']}")
        print(f"  Note: {res['note']}")
        
        # Full tests with column names (no limit to 3)
        mar_count = sum(1 for p in res['p_values'] if not np.isnan(p) and p <= alpha)
        print(f"  Summary of Tests: {len([p for p in res['p_values'] if not np.isnan(p)])} tests run, {mar_count} indicate MAR")
        print("  All Decisions (with column names):")
        for i, (p_val, decision, other_col) in enumerate(zip(res['p_values'], res['decisions'], complete_cols)):
            if i >= len(complete_cols):
                break
            original_col = other_col.replace('_numeric', '')
            if not np.isnan(p_val):
                p_str = f"{p_val:.3f}" if p_val > 0.001 else f"{p_val:.2e}"
                print(f"    - vs. {original_col}: p={p_str}, {decision}")
            else:
                print(f"    - vs. {original_col}: {decision}")
    
    # Summary table to avoid truncation
    summary_df = pd.DataFrame({
        'Column': list(results.keys()),
        'Missing Rate': [res['missing_rate'] for res in results.values()],
        'Overall Decision': [res['overall_decision'] for res in results.values()],
        'MAR Tests': [sum(1 for p in res['p_values'] if not np.isnan(p) and p <= alpha) for res in results.values()]
    })
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    return results

def visualize_missingness(df, figsize=(12, 6)):
    """
    Visualize missingness patterns in the DataFrame.
    
    Parameters:
    - df: DataFrame.
    - figsize: Tuple for figure size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Missingness heatmap
    sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis', cbar_kws={'label': 'Missing (1)'}, ax=ax1)
    ax1.set_title('Missingness Heatmap')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Rows')
    
    # Missing rate bar plot
    missing_rates = df.isnull().mean().sort_values(ascending=False)
    missing_rates.plot(kind='bar', ax=ax2)
    ax2.set_title('Missing Rates by Column')
    ax2.set_ylabel('Missing Proportion')
    ax2.set_xlabel('Columns')
    
    plt.tight_layout()
    plt.show()

# Demo if run as script (unchanged)
if __name__ == "__main__":
    # ... (previous demo code remains the same)
    pass

# Simulation and introduction functions (unchanged from previous version)
def simulate_data(n_samples=1000, n_features=5, seed=42):
    """Simulate a dataset with features and a target variable."""
    np.random.seed(seed)
    data = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    data['target'] = data['feature_0'] * 2 + data['feature_1'] * 1.5 + np.random.normal(0, 0.5, n_samples)
    return data

def introduce_mcar(df, columns, missing_rate=0.1, seed=42):
    """Introduce Missing Completely At Random (MCAR) missingness."""
    df_mcar = df.copy()
    np.random.seed(seed)
    mask = np.random.rand(len(df), len(columns)) < missing_rate
    for i, col in enumerate(columns):
        df_mcar.loc[:, col][mask[:, i]] = np.nan
    return df_mcar

def introduce_mar(df, missing_col, predictor_cols, missing_rate=0.1, seed=42):
    """Introduce Missing At Random (MAR) missingness."""
    df_mar = df.copy()
    np.random.seed(seed)
    probs = np.zeros(len(df))
    for pred_col in predictor_cols:
        probs += stats.norm.cdf(df[pred_col])
    probs = probs / len(predictor_cols)
    mask = np.random.rand(len(df)) < (missing_rate * probs)
    df_mar.loc[mask, missing_col] = np.nan
    return df_mar

def introduce_mnar(df, missing_col, threshold=0, missing_rate=0.1, seed=42):
    """Introduce Missing Not At Random (MNAR) missingness."""
    df_mnar = df.copy()
    np.random.seed(seed)
    probs = np.where(df[missing_col] < threshold, missing_rate, 0)
    mask = np.random.rand(len(df)) < probs
    df_mnar.loc[mask, missing_col] = np.nan
    return df_mnar