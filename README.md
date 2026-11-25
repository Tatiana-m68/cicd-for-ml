# CI for Titanic Survival Prediction

This project trains a Titanic survival classifier and automatically checks data and model quality using GitHub Actions.

## Data and model

- Source: [seaborn-data Titanic dataset](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).
- Split: stratified 80/20, with 712 training rows and 179 test rows.
- Features: `pclass`, `sex`, `age`, `fare`.
- Target: `survived` (`0` = did not survive, `1` = survived).

The model is a scikit-learn pipeline with median imputation for numeric features, one-hot encoding for `sex`, and a Random Forest classifier. Preprocessing and model fitting use only the training data.

## Project structure

| Path | Purpose |
| --- | --- |
| `data/` | Training and test CSV files |
| `src/train.py` | Train the pipeline and save `model.joblib` |
| `tests/` | Eight data and model tests |
| `.github/workflows/ci.yml` | GitHub Actions workflow |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Exclude generated and temporary files from Git |

## Run locally

Use Python 3.11. Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python src/train.py
python -m pytest -q
```

Training creates `model.joblib`. This generated file is excluded from Git and is recreated during CI.

## Automated checks

The eight tests check:

- Required columns, a non-missing binary target, and non-negative fares.
- Test accuracy of at least 0.75 and accuracy above the test-set majority-class baseline.
- One prediction per test row and valid predictions for raw passenger input with a missing age.

On every push and pull request, GitHub Actions installs dependencies, trains the model, and runs the tests. A failed command or test makes the workflow fail.

[View workflow runs](https://github.com/Tatiana-m68/cicd-for-ml/actions).

## Exercise notebook

The [completed notebook](https://colab.research.google.com/drive/1_p3b00PFkBUQCECBnGVAy6PkDLqSI2aE) includes data preparation, an intentional test failure caused by a negative fare, and successful checks after restoring the original data.
