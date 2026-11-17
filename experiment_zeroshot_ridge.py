"""Compatibility entry point: Ridge on mean_pooled versus both zero-shot scores.

Uses the shared validated loading, nested position CV and Spearman/RMSE/MAE.
For all six representations run: python -m tem1.supervised
"""
from tem1.supervised import main

if __name__ == '__main__':
    raise SystemExit(main(default_representations=('mean_pooled',),
                          default_output='results/ridge_mean_pooled'))
