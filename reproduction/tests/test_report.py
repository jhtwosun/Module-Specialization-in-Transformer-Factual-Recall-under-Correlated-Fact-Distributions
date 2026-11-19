"""Host-safe report checks: no torch, NumPy, matplotlib or training."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction import report


class ReportTests(unittest.TestCase):
    def fixture(self, root, seeds=(0, 1, 2), bad=None):
        root = Path(root)
        (root / 'source').mkdir()
        (root / 'source' / 'code.txt').write_text('fixed source')
        digest = hashlib.sha256(b'fixed source').hexdigest()
        cells = []
        for seed in seeds:
            cfg = dict(seed=seed, motif='subject', collision=2, layers=2,
                       d_model=128, probe_mode='joint', steps=8000)
            cp = root / f'config{seed}.json'
            cp.write_text(json.dumps(cfg))
            out = root / f'cell{seed}'
            out.mkdir()
            (out / 'config.json').write_text(json.dumps(cfg))
            status = dict(returncode=1 if seed == bad else 0, finished_utc='2026-01-01')
            (out / 'execution-123.json').write_text(json.dumps(status))
            metrics = dict(accuracy=.9, micro_accuracy=.95, probe_macro=[.1,.5,.8,.9],
                           mean_ablation_macro=[.2,.6,.8,.7],
                           patching={'module_0':{'effect_pp':80+seed},
                                     'strict_edge_0_1':{'effect_pp':30+seed},
                                     'legacy_edge_0_1':{'effect_pp':40+seed}})
            (out / 'metrics.json').write_text(json.dumps(metrics))
            cells.append(dict(cell=str(seed), family='synthetic', config=cp.name,
                              config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(), output=out.name))
        (root / 'manifest.json').write_text(json.dumps(dict(source_sha256={'code.txt':digest},cells=cells)))
        return root

    def test_seed_aggregation_preserves_conventions_and_units(self):
        with tempfile.TemporaryDirectory() as d:
            records, coverage, configs = report.load_runs([self.fixture(d)])
            rows, gaps = report.aggregate(records, coverage)
            self.assertFalse(gaps)
            self.assertEqual(len(configs), 1)
            values = {r['metric']:r for r in rows}
            self.assertAlmostEqual(values['delta2_macro_pp']['mean'], 10)
            self.assertEqual(values['patching.strict_edge_0_1.effect_pp']['mean'], 31)
            self.assertEqual(values['patching.legacy_edge_0_1.effect_pp']['mean'], 41)
            self.assertEqual(values['patching.strict_edge_0_1.effect_pp']['std_sample'], 1)
            self.assertAlmostEqual(values['mean_ablation_macro_change_pp.0']['mean'], -70)
            self.assertNotIn('mean_ablation_micro_change_pp.0', values)

    def test_failed_seed_never_contributes_stale_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            records, coverage, _ = report.load_runs([self.fixture(d, bad=1)])
            rows, gaps = report.aggregate(records, coverage)
            self.assertEqual([r['seed'] for r in records], [0,2])
            self.assertTrue(gaps)
            self.assertFalse(any(r['complete'] for r in rows))

    def test_missing_seed_is_gap_and_sample_std_undefined(self):
        with tempfile.TemporaryDirectory() as d:
            records, coverage, _ = report.load_runs([self.fixture(d, seeds=(0,))])
            rows, gaps = report.aggregate(records, coverage)
            self.assertEqual(gaps[0]['status'], 'seed coverage mismatch')
            self.assertTrue(all(r['std_sample'] is None for r in rows))

    def test_config_and_snapshot_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            (root/'source/code.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Source hash'):
                report.load_runs([root])
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            (root/'config0.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Config hash'):
                report.load_runs([root])

    def test_resolved_config_mismatch_is_gap(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            (root/'cell0/config.json').write_text('{}')
            records, coverage, _ = report.load_runs([root])
            self.assertEqual(len(records), 2)
            self.assertIn('differs', coverage[0]['status'])

    def test_duplicates_rejected_instead_of_counted_as_extra_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                report.load_runs([root, root])

    def test_edge_ids_and_missing_metrics(self):
        value = {'layers':[{'source':2,'receiver':4,'edge_effect_pp':.5},
                           {'source':0,'receiver':4,'edge_effect_pp':1.5}]}
        f = report.flatten(value)
        self.assertEqual(f['layers.2->4.edge_effect_pp'], .5)
        self.assertNotIn('layers.2->4.source', f)
        records = [dict(group='g',seed=0,metrics={'a':1,'b':3}), dict(group='g',seed=1,metrics={'a':2})]
        cov = [dict(group='g',status='complete')]
        rows,gaps = report.aggregate(records,cov,(0,1))
        self.assertTrue(gaps)
        self.assertFalse(next(r for r in rows if r['metric']=='b')['complete'])
        with self.assertRaises(ValueError):
            report.flatten({'bad':float('nan')})

    def test_table_export_has_seven_tables_and_no_reference_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'run'; root.mkdir()
            records,cov,configs = report.load_runs([self.fixture(root)])
            rows,gaps = report.aggregate(records,cov)
            dest = Path(d)/'report'
            report.emit(dest,records,cov,configs,rows,gaps)
            self.assertEqual(len(list(dest.glob('table_*.csv'))),7)
            self.assertIn('MISSING:', (dest/'tables.md').read_text())
            self.assertTrue((dest/'per_seed.csv').is_file())

    def test_plot_guard_precedes_matplotlib_import(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'Slurm'):
                report.plot('/tmp/not-created', [], {})

    def test_different_training_configs_do_not_pool(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            manifest = report.read(root/'manifest.json')
            cfg = report.read(root/'config1.json'); cfg['steps'] = 100
            (root/'config1.json').write_text(json.dumps(cfg))
            (root/'cell1/config.json').write_text(json.dumps(cfg))
            manifest['cells'][1]['config_sha256'] = hashlib.sha256((root/'config1.json').read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps(manifest))
            records,cov,configs = report.load_runs([root])
            self.assertEqual(len(configs),2)
            _,gaps = report.aggregate(records,cov)
            self.assertEqual(len(gaps),2)

    def test_pretrained_natural_subcells_and_selection_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d,seeds=(0,))
            cfg = dict(experiment='natural_counterfact',model='test',seeds=[0,1,2],relations=['P30'])
            (root/'config0.json').write_text(json.dumps(cfg))
            (root/'cell0/config.json').write_text(json.dumps(cfg))
            manifest = report.read(root/'manifest.json')
            manifest['cells'][0].update(family='pretrained',config_sha256=hashlib.sha256((root/'config0.json').read_bytes()).hexdigest())
            (root/'manifest.json').write_text(json.dumps(manifest))
            summary = [dict(seed=s,relation='P30',layers=[dict(source=2,receiver=4,edge_module_ratio=.3)],
                            maximum=dict(source=2,receiver=4,edge_module_ratio=.3)) for s in (0,1,2)]
            (root/'cell0/summary.json').write_text(json.dumps(summary))
            (root/'cell0/resolved_model.json').write_text(json.dumps(dict(commit='abc')))
            records,cov,configs = report.load_runs([root])
            rows,gaps = report.aggregate(records,cov)
            self.assertFalse(gaps)
            self.assertEqual(len(report.select_table('pythia-natural',rows,configs)),2)
            self.assertEqual(next(iter(configs.values()))['config']['resolved_model']['commit'],'abc')


if __name__ == '__main__':
    unittest.main()
