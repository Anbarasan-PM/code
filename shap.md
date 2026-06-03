If you're working in Jupyter Notebook, I'd recommend a **single end-to-end workflow** that produces:

1. SHAP Impact Categories
2. Green vs Amber comparison
3. Feature ranking
4. Statistical significance
5. Amber-driving category combinations
6. Executive summary tables

---

# Cell 1: Imports

```python
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu
from itertools import combinations

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
```

---

# Cell 2: Define Features

Replace with your feature names.

```python
features = [
    'feature_1',
    'feature_2',
    'feature_3',
    'feature_4',
    'feature_5',
    'feature_6',
    'feature_7'
]

target = 'cod_band'
```

---

# Cell 3: Keep Only Green and Amber

We are interested in the deterioration step.

```python
df_ga = df[df[target].isin(['Green', 'Amber'])].copy()

print(df_ga[target].value_counts())
```

---

# Cell 4: Create SHAP Impact Categories

### Why?

Convert:

```text
-0.823
-0.311
0.015
0.442
```

into:

```text
Strong Negative
Moderate Negative
Neutral
Moderate Positive
Strong Positive
```

which is much easier to analyze.

```python
def create_shap_category(shap_series):

    p10 = shap_series.quantile(0.10)
    p30 = shap_series.quantile(0.30)
    p70 = shap_series.quantile(0.70)
    p90 = shap_series.quantile(0.90)

    def categorize(x):

        if x <= p10:
            return "Strong Negative"

        elif x <= p30:
            return "Moderate Negative"

        elif x <= p70:
            return "Neutral"

        elif x <= p90:
            return "Moderate Positive"

        else:
            return "Strong Positive"

    return shap_series.apply(categorize)
```

---

# Cell 5: Apply Categories to All Features

Assuming SHAP columns are:

```text
feature_1_shap
feature_2_shap
...
```

```python
for feature in features:

    shap_col = f"{feature}_shap"

    impact_col = f"{feature}_impact"

    df_ga[impact_col] = create_shap_category(df_ga[shap_col])
```

---

# Cell 6: Verify Categories

```python
impact_cols = [f"{f}_impact" for f in features]

df_ga[impact_cols].head()
```

---

# Cell 7: Compare SHAP Distributions

This is your primary driver analysis.

```python
green = df_ga[df_ga[target] == 'Green']
amber = df_ga[df_ga[target] == 'Amber']
```

```python
feature_results = []

for feature in features:

    shap_col = f"{feature}_shap"

    g = green[shap_col].dropna()
    a = amber[shap_col].dropna()

    stat, pvalue = mannwhitneyu(
        g,
        a,
        alternative='two-sided'
    )

    green_mean = g.mean()
    amber_mean = a.mean()

    shap_shift = amber_mean - green_mean

    feature_results.append({
        "feature": feature,
        "green_mean": green_mean,
        "amber_mean": amber_mean,
        "shap_shift": shap_shift,
        "pvalue": pvalue
    })

feature_shift_df = pd.DataFrame(feature_results)
```

---

# Cell 8: Calculate Category Shift

Assign scores to categories.

```python
impact_score = {
    "Strong Negative": -2,
    "Moderate Negative": -1,
    "Neutral": 0,
    "Moderate Positive": 1,
    "Strong Positive": 2
}
```

```python
category_results = []

for feature in features:

    impact_col = f"{feature}_impact"

    green_score = (
        green[impact_col]
        .map(impact_score)
        .mean()
    )

    amber_score = (
        amber[impact_col]
        .map(impact_score)
        .mean()
    )

    category_shift = amber_score - green_score

    category_results.append({
        "feature": feature,
        "green_category_score": green_score,
        "amber_category_score": amber_score,
        "category_shift": category_shift
    })

category_shift_df = pd.DataFrame(category_results)
```

---

# Cell 9: Create Final Driver Ranking

This identifies variables driving Green → Amber.

```python
driver_df = feature_shift_df.merge(
    category_shift_df,
    on='feature'
)
```

```python
driver_df['driver_score'] = (
    abs(driver_df['shap_shift'])
    *
    abs(driver_df['category_shift'])
    *
    (-np.log10(driver_df['pvalue'] + 1e-10))
)
```

```python
driver_df = driver_df.sort_values(
    'driver_score',
    ascending=False
)

driver_df
```

---

# Cell 10: Impact Category Distribution

Shows how category frequencies differ.

```python
distribution_results = []

for feature in features:

    impact_col = f"{feature}_impact"

    temp = pd.crosstab(
        df_ga[impact_col],
        df_ga[target],
        normalize='columns'
    )

    temp = temp.reset_index()

    temp['feature'] = feature

    distribution_results.append(temp)

impact_distribution = pd.concat(
    distribution_results,
    ignore_index=True
)

impact_distribution.head()
```

---

# Cell 11: Amber Lift Analysis

This identifies which SHAP category is most associated with Amber.

```python
lift_results = []

for feature in features:

    impact_col = f"{feature}_impact"

    for category in df_ga[impact_col].unique():

        amber_pct = (
            amber[impact_col]
            .eq(category)
            .mean()
        )

        green_pct = (
            green[impact_col]
            .eq(category)
            .mean()
        )

        lift = amber_pct / max(green_pct, 0.0001)

        lift_results.append({
            "feature": feature,
            "category": category,
            "green_pct": green_pct,
            "amber_pct": amber_pct,
            "lift": lift
        })

lift_df = pd.DataFrame(lift_results)

lift_df.sort_values(
    "lift",
    ascending=False
).head(20)
```

---

# Cell 12: Find Strong Amber Combinations

This is usually where the biggest insights come from.

### Pairwise Feature Combinations

```python
combination_results = []

impact_cols = [f"{f}_impact" for f in features]
```

```python
for f1, f2 in combinations(features, 2):

    col1 = f"{f1}_impact"
    col2 = f"{f2}_impact"

    temp = df_ga.copy()

    temp["combo"] = (
        temp[col1]
        + " | "
        + temp[col2]
    )

    green_freq = (
        temp[temp[target]=="Green"]
        ["combo"]
        .value_counts(normalize=True)
    )

    amber_freq = (
        temp[temp[target]=="Amber"]
        ["combo"]
        .value_counts(normalize=True)
    )

    combo_df = pd.concat(
        [green_freq, amber_freq],
        axis=1
    )

    combo_df.columns = [
        'green_pct',
        'amber_pct'
    ]

    combo_df = combo_df.fillna(0)

    combo_df["lift"] = (
        combo_df["amber_pct"]
        /
        combo_df["green_pct"].replace(0,0.0001)
    )

    combo_df = combo_df.reset_index()

    combo_df["feature_pair"] = f"{f1} + {f2}"

    combination_results.append(combo_df)
```

```python
combination_df = pd.concat(
    combination_results,
    ignore_index=True
)
```

---

# Cell 13: Top Amber Signatures

```python
top_amber_signatures = (
    combination_df
    .query("amber_pct > 0.05")
    .sort_values(
        "lift",
        ascending=False
    )
)

top_amber_signatures.head(20)
```

---

# Cell 14: Executive Summary Table

The table you will present.

```python
executive_summary = driver_df[
    [
        'feature',
        'green_mean',
        'amber_mean',
        'shap_shift',
        'category_shift',
        'pvalue',
        'driver_score'
    ]
]

executive_summary
```

---

# Interpretation Guide

After running the notebook:

### Driver Ranking

Look at:

```python
driver_df
```

Top-ranked features are the strongest Green → Amber deterioration drivers.

---

### Amber Categories

Look at:

```python
lift_df
```

Example:

| Feature     | Category        | Lift |
| ----------- | --------------- | ---- |
| Utilization | Strong Negative | 8.1  |

Interpretation:

> Records with Strong Negative SHAP impact from Utilization are 8.1× more likely to be Amber than Green.

---

### Amber Signatures

Look at:

```python
top_amber_signatures
```

Example:

| Combination               | Lift |
| ------------------------- | ---- |
| Utilization=SN + Delay=SN | 12.4 |

Interpretation:

> The simultaneous negative impact of Utilization and Delay is a major signature of Green → Amber transition.

This notebook gives you both **individual drivers** and **interaction patterns**, which is usually what stakeholders want when asking *"what caused records to move from Green to Amber?"* rather than just looking at SHAP plots.













--------------------#################


The **driver score** in the notebook is **not a standard SHAP metric**. It's a custom ranking score that combines three things:

1. **How much the SHAP contribution changed** (Green → Amber)
2. **How much the impact category changed** (Positive → Negative)
3. **How statistically significant the difference is**

The code was:

```python
driver_df['driver_score'] = (
    abs(driver_df['shap_shift'])
    *
    abs(driver_df['category_shift'])
    *
    (-np.log10(driver_df['pvalue'] + 1e-10))
)
```

---

## What each component means

### 1. SHAP Shift

```python
shap_shift = amber_mean_shap - green_mean_shap
```

Example:

| Feature     | Green Mean SHAP | Amber Mean SHAP | Shift |
| ----------- | --------------- | --------------- | ----- |
| Utilization | 0.40            | -0.30           | -0.70 |

Interpretation:

> The feature's contribution became much more negative in Amber.

A large negative shift is a strong signal that the feature is pushing records away from Green.

---

### 2. Category Shift

Suppose we mapped:

| Category          | Score |
| ----------------- | ----- |
| Strong Negative   | -2    |
| Moderate Negative | -1    |
| Neutral           | 0     |
| Moderate Positive | 1     |
| Strong Positive   | 2     |

Then:

| Feature     | Green Avg Score | Amber Avg Score |
| ----------- | --------------- | --------------- |
| Utilization | 1.6             | -1.2            |

```python
category_shift = -1.2 - 1.6 = -2.8
```

Interpretation:

> The feature moved from generally positive impact in Green to generally negative impact in Amber.

---

### 3. Statistical Significance

Using:

```python
mannwhitneyu()
```

You get a p-value.

| P-value | Meaning               |
| ------- | --------------------- |
| 0.50    | No real difference    |
| 0.05    | Significant           |
| 0.001   | Very significant      |
| <0.0001 | Extremely significant |

The transformation:

```python
-np.log10(pvalue)
```

gives:

| P-value | Weight |
| ------- | ------ |
| 0.1     | 1      |
| 0.01    | 2      |
| 0.001   | 3      |
| 0.0001  | 4      |

Smaller p-values receive larger weights.

---

## Example

Feature A:

```python
shap_shift = -0.60
category_shift = -2.5
pvalue = 0.0001
```

Driver score:

```python
0.60 × 2.5 × 4
= 6.0
```

Feature B:

```python
shap_shift = -0.20
category_shift = -0.5
pvalue = 0.01
```

Driver score:

```python
0.20 × 0.5 × 2
= 0.2
```

Feature A ranks much higher.

---

## But here's what I would recommend

For explainability projects, I usually avoid creating a synthetic "driver score" because it can be hard to justify to stakeholders.

Instead, rank features by a table like:

| Rank | Feature   | SHAP Shift | P-value | Amber Strong Negative % | Green Strong Negative % |
| ---- | --------- | ---------- | ------- | ----------------------- | ----------------------- |
| 1    | Feature A | -0.62      | <0.001  | 34%                     | 6%                      |
| 2    | Feature B | -0.48      | <0.001  | 28%                     | 8%                      |

This is easier to explain:

> "Feature A became substantially more negative in Amber, and Strong Negative impacts occur 34% of the time in Amber versus only 6% in Green."

For business audiences, this is usually more convincing than a custom driver score.

In fact, for your Green → Amber analysis, I'd focus on three outputs:

### A. Feature Deterioration Table

| Feature | Green Mean SHAP | Amber Mean SHAP | SHAP Shift |
| ------- | --------------- | --------------- | ---------- |

### B. Impact Category Shift

| Feature | Green Strong Neg % | Amber Strong Neg % | Increase |
| ------- | ------------------ | ------------------ | -------- |

### C. Amber Signatures

| Combination | Amber % | Green % | Lift |
| ----------- | ------- | ------- | ---- |

These directly answer **which variables and combinations are causing the shift from Green to Amber** without relying on an invented composite score.

