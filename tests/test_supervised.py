"""Identity, grouping, nested selection and OOF evaluation regression checks."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from tem1.supervised import (prepare_tables, fit_outer_fold, benchmark, metric_values,
                             summarize, VECTOR_COLUMNS)


class SupervisedTests(unittest.TestCase):
    def setUp(self):
        reference = 'ACDEFGHIKL'
        self.reference = reference
        self.folds = pd.DataFrame([
            dict(mutant=f'{aa}{p+1}{alt}', position=p+1, wt_aa=aa, mut_aa=alt,
                 fold=p % 5, DMS_score=float(p) + j / 5)
            for p, aa in enumerate(reference) for j, alt in enumerate('MN')])
        self.raw = self.folds[['mutant', 'DMS_score']].copy()
        self.raw['mutated_sequence'] = [reference[:r.position-1] + r.mut_aa + reference[r.position:]
                                        for r in self.folds.itertuples()]
        self.masked = self.folds.rename(columns={'wt_aa':'wild_type','mut_aa':'alternate'}).copy()
        self.masked['esm2_score'] = self.folds['DMS_score'] * 2
        self.features = self.folds[['mutant','position','fold']].copy()
        rng = np.random.default_rng(42)
        self.features['zeroshot'] = np.linspace(-1, 1, len(self.folds))
        for column in VECTOR_COLUMNS:
            self.features[column] = list(rng.normal(size=(len(self.folds), 4)).astype(np.float16))
        self.hidden = rng.normal(size=(10, 4))
        self.logits = np.zeros((10, 33))
        aa_ids = {aa: i+4 for i,aa in enumerate('LAGVSERTIDPKQNFYMHWC')}
        for row,z in zip(self.folds.itertuples(),self.features['zeroshot']):
            self.logits[row.position-1,aa_ids[row.mut_aa]] = z

    def prepare(self, **changes):
        tables = dict(folds=self.folds, features=self.features, raw=self.raw,
                      masked=self.masked, reference=self.reference,
                      wt_hidden=self.hidden, wt_logits=self.logits, expected_count=20, hidden_size=4)
        tables.update(changes)
        return prepare_tables(**tables)

    def test_all_tables_reordered_by_mutant(self):
        prepared = self.prepare(features=self.features.sample(frac=1,random_state=1),
                                raw=self.raw.sample(frac=1,random_state=2),
                                masked=self.masked.sample(frac=1,random_state=3))
        self.assertEqual(prepared.rows.mutant.tolist(), self.folds.mutant.tolist())
        np.testing.assert_array_equal(prepared.rows.zeroshot_wt, self.features.zeroshot)
        np.testing.assert_array_equal(prepared.representation('emb_mut_pos_delta')[:,4:],
            prepared.matrices['emb_mut_pos']-self.hidden[self.folds.position.to_numpy()-1])

    def test_global_representation_extends_position_and_delta_blocks(self):
        prepared = self.prepare(features=self.features.sample(frac=1, random_state=11))
        paired = prepared.representation('emb_mut_pos_delta')
        extended = prepared.representation('emb_mut_pos_delta_mean_pooled')
        self.assertEqual(extended.shape, (20, 12))  # Three blocks of hidden size 4.
        np.testing.assert_array_equal(extended[:, :8], paired)
        np.testing.assert_array_equal(extended[:, 8:], np.stack(self.features['mean_pooled']))

    def test_missing_duplicate_and_nonfinite_rejected_without_filtering(self):
        cases = [self.features.iloc[:-1], pd.concat([self.features,self.features.iloc[:1]])]
        bad = self.features.copy(); bad.loc[0,'zeroshot'] = np.nan; cases.append(bad)
        bad = self.features.copy(); bad['DMS_score'] = self.folds.DMS_score; bad.loc[0,'DMS_score'] = np.nan; cases.append(bad)
        for features in cases:
            with self.subTest(), self.assertRaises(ValueError):self.prepare(features=features)

    def test_target_position_fold_and_sequence_mismatches_rejected(self):
        for name,column,value in [('raw','DMS_score',99.),('masked','DMS_score',99.),
                                  ('features','position',2),('features','fold',1),
                                  ('raw','mutated_sequence','ACDEFGHIKL')]:
            table = getattr(self,name).copy();table.loc[0,column]=value
            with self.subTest(name=name,column=column), self.assertRaises(ValueError):
                self.prepare(**{name:table})
        folds=self.folds.copy();folds.loc[0,'fold']=1
        with self.assertRaisesRegex(ValueError,'more than one fold'):self.prepare(folds=folds)

    def test_noninteger_positions_and_malformed_embeddings_rejected(self):
        folds=self.folds.copy();folds['position']=folds['position'].astype(float);folds.loc[0,'position']=1.5
        with self.assertRaises(ValueError):self.prepare(folds=folds)
        features=self.features.copy();features.at[0,'emb_mut_pos']=np.zeros(3)
        with self.assertRaises(ValueError):self.prepare(features=features)

    def test_outer_test_targets_cannot_affect_alpha_or_predictions(self):
        data=self.prepare();X=data.representation('emb_mut_pos')
        y=data.rows.DMS_score.to_numpy();positions=data.rows.position.to_numpy();folds=data.rows.fold.to_numpy()
        with threadpool_limits(limits=1):
            test,pred,report=fit_outer_fold(X,y,positions,folds,0,[.1,10.])
            changed=y.copy();changed[test]=100 + np.arange(test.sum())
            _,pred_changed,report_changed=fit_outer_fold(X,changed,positions,folds,0,[.1,10.])
        np.testing.assert_array_equal(pred,pred_changed)
        self.assertEqual(report['alpha'],report_changed['alpha'])
        self.assertEqual(report['inner_mse_by_alpha'],report_changed['inner_mse_by_alpha'])
        self.assertAlmostEqual(report['selected_inner_mse'],min(r['mean_mse'] for r in report['inner_mse_by_alpha']))
        self.assertEqual(len(report['inner_splits']),4)

    def test_scaler_fits_only_inner_training_rows_then_outer_training(self):
        data = self.prepare()
        X = data.representation('emb_mut_pos').copy()
        X[:, 0] = np.arange(len(X))  # Unique row marker, visible in scaler.fit.
        folds = data.rows.fold.to_numpy()
        seen = []

        class RecordingScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                seen.append(set(X[:, 0]))
                return super().fit(X, y, sample_weight=sample_weight)

        with patch('tem1.supervised.StandardScaler', RecordingScaler), threadpool_limits(limits=1):
            fit_outer_fold(X, data.rows.DMS_score.to_numpy(), data.rows.position.to_numpy(),
                           folds, 0, [.1, 10.])
        outer_train = set(np.flatnonzero(folds != 0))
        self.assertEqual(len(seen), 9)  # 2 alphas x 4 inner fits, plus outer refit.
        self.assertTrue(all(rows < outer_train for rows in seen[:-1]))
        self.assertEqual(seen[-1], outer_train)

    def test_oof_complete_and_summary_separates_fold_mean_from_pooled(self):
        data=self.prepare()
        predictions,reports,_=benchmark(data,('mean_pooled','emb_mut_pos_delta_mean_pooled'),[.1,10.],1)
        self.assertEqual(predictions.mutant.tolist(),self.folds.mutant.tolist())
        self.assertTrue(np.isfinite(predictions.select_dtypes('number')).all().all())
        for method,report in reports.items():
            regression=report['kind']=='ridge'
            self.assertEqual(metric_values(predictions.DMS_score,predictions[method],regression),report['oof_metrics'])
            fold_spearman=[r['metrics']['spearman'] for r in report['folds']]
            self.assertEqual(report['fold_summary']['spearman']['mean'],np.mean(fold_spearman))
            self.assertEqual(report['fold_summary']['spearman']['std'],np.std(fold_spearman,ddof=1))
            self.assertNotIn('pearson',str(report).lower())
            if not regression:self.assertEqual(set(report['oof_metrics']),{'spearman'})
