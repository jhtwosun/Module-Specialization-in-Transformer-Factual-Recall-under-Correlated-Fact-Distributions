"""Lightweight stdlib checks. Safe on the login node; no ML imports."""
import json
from pathlib import Path
import tempfile
import unittest

from reproduction.matrix import synthetic_cells, pretrained_cells
from reproduction.launch import submission_command


class MatrixTests(unittest.TestCase):
    def test_combined_complete_cartesian_grid(self):
        cells=synthetic_cells("combined")
        self.assertEqual(len(cells),7*8*3)
        self.assertEqual(len({(c['collision'],c['answer_pool'],c['seed']) for c in cells}),168)

    def test_explicit_capacity_ambiguities(self):
        cells=synthetic_cells("capacity")
        for d in (64,128,256):
            group=[c for c in cells if c['d_model']==d]
            self.assertTrue({2,4} <= {c['collision'] for c in group if c['motif']=='relation'})
            self.assertTrue({4,8} <= {c['answer_pool'] for c in group if c['motif']=='answer'})

    def test_paper_probe_optimizer(self):
        for c in synthetic_cells("hierarchy"):
            self.assertEqual(c['lr'],c['probe_lr'])
            self.assertEqual(c['weight_decay'],c['probe_weight_decay'])
            self.assertFalse(c['legacy_repeat_relation'])

    def test_answer_axis_reaches_sampler(self):
        for family in ('hierarchy', 'capacity', 'causal', 'threshold', 'frozen', 'scale'):
            for c in synthetic_cells(family):
                if c['motif'] == 'answer':
                    self.assertEqual(c['collision'], c['answer_pool'])

    def test_single_axis_and_combined_sampling(self):
        self.assertTrue(all(c['block_sampling'] == 'legacy_without_replacement'
                            for c in synthetic_cells('hierarchy')))
        self.assertTrue(all(c['block_sampling'] == 'with_replacement'
                            for c in synthetic_cells('combined')))

    def test_natural_preserves_pairs_per_fact(self):
        with self.assertRaises(ValueError):
            pretrained_cells("natural")
        cells=pretrained_cells("natural","/tmp/counterfact.json")
        self.assertEqual(len(cells),36)
        # Each cell is capped at 100 facts and 20 pairs per fact.
        self.assertTrue(all(c['pairs_per_fact']==20 for c in cells))
        self.assertTrue(all(c['max_facts']==100 for c in cells))
        self.assertTrue(all(c['max_pairs_total']==2000 for c in cells))

    def test_bounded_batch_submission(self):
        manifest={'cells':[{},{}]}
        cmd=submission_command(Path('/tmp/run'),manifest,Path('/tmp/image'))
        self.assertIn('--partition=backfill',cmd)
        self.assertIn('--array=0-1%2',cmd)
        self.assertEqual(cmd[0],'sbatch')
        with self.assertRaises(ValueError):
            submission_command(Path('/tmp/run'),manifest,Path('/tmp/image'),concurrency=100)


if __name__=='__main__':
    unittest.main()
