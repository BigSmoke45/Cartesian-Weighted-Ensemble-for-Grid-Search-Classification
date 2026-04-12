# Cartesian Weighted Ensemble for Grid Search Classification

A weighted soft-voting ensemble combining `DecisionTreeClassifier` and `LogisticRegression`, evaluated over multiple training runs to assess stability and accuracy variance.

---

## What it does

- Trains a `VotingClassifier` (soft voting, weights `[0.6, 0.4]`) repeatedly over 12 independent runs
- Measures accuracy variance across runs to assess model stability
- Finds accuracy peaks across runs using `scipy.signal.find_peaks`
- Plots accuracy per run with peak markers highlighted

---

## Ensemble Setup

```python
VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier()),
        ('lr', Pipeline([SimpleImputer(), LogisticRegression(solver='liblinear')]))
    ],
    voting='soft',
    weights=[0.6, 0.4]
)
```

---

## Output

- Mean accuracy across 12 runs
- Standard deviation (stability metric)
- Accuracy plot with peak run markers

---

## Tech Stack

`Python` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `SciPy`

---

## Usage

```bash
pip install scikit-learn pandas numpy matplotlib scipy
# Replace file_path in main() with your CSV path
python cartesian_weighted_ensemble.py
```

---

## Notes

University coursework project. Dataset used: Animal Crossing item catalog (socks.csv). The pipeline is dataset-agnostic — swap in any CSV with categorical features.
