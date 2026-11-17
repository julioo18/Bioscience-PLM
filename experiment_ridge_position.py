"""Compatibility entry point: Ridge on emb_mut_pos plus WT-relative delta (1280).

Uses the shared validated loading, nested position CV and Spearman/RMSE/MAE.
For all six representations run: python -m tem1.supervised
"""
from tem1.supervised import main

if __name__ == '__main__':
    raise SystemExit(main(default_representations=('emb_mut_pos_delta',),
                          default_output='results/ridge_position'))
