"""Login-safe rebuttal matrix and entropy contracts."""
import unittest
from reproduction.rebuttal.matrix import cells
from reproduction.rebuttal.counterfact import profile
from reproduction.synthetic.run import resolve
from reproduction.report import subcells


def fact(i,s,a,r='P1'):
    return dict(case_id=i,requested_rewrite=dict(subject=s,relation_id=r,target_true=dict(id=a,str=a)))


class RebuttalTests(unittest.TestCase):
    def test_matrix_covers_reported_axes_and_seeds(self):
        self.assertEqual(len(cells('decay')),12)
        self.assertEqual({c['weight_decay'] for c in cells('decay')},{0.,.001,.01,.1})
        self.assertEqual({c['value_init'] for c in cells('init')},{'small_normal','scaled_identity','orthogonal'})
        self.assertEqual({c['correlation'] for c in cells('correlation')},{.5,.7,.9,1.})
        self.assertEqual(len(cells('natural',dataset_path='/tmp/counterfact.json')),72)
        self.assertEqual(len(cells('twohop',tuple(range(8)))),24)
        for family in ('decay','init','correlation','relation','joint','answer_convergence'):
            for cfg in cells(family):resolve(cfg)

    def test_invalid_initialization_and_schedule(self):
        for cfg in ({'value_init':'unknown'},{'identity_noise':-1},{'value_init_std':float('nan')},{'warmup_steps':-1}):
            with self.assertRaises(ValueError):resolve(cfg)

    def test_conditional_entropy_distinguishes_repeated_and_unique_subjects(self):
        independent=profile([fact(0,'s0','a0'),fact(1,'s0','a1'),fact(2,'s1','a0'),fact(3,'s1','a1')])[0]
        self.assertAlmostEqual(independent['entropy_bits'],1.)
        self.assertAlmostEqual(independent['conditional_entropy_given_subject_bits'],1.)
        unique=profile([fact(0,'s0','a0'),fact(1,'s1','a1')])[0]
        self.assertEqual(unique['conditional_entropy_given_subject_bits'],0.)
        self.assertEqual(unique['singleton_subject_fraction'],1.)
        # Low H(A|S) here is a sampling property, not evidence of clustering.
        self.assertEqual(unique['entropy_bits'],1.)

    def test_single_answer_normalization_and_duplicate_ids(self):
        self.assertEqual(profile([fact(0,'s','a')])[0]['normalized_by_observed_answers'],0.)
        with self.assertRaises(ValueError):profile([fact(0,'s','a'),fact(0,'t','b')])

    def test_generic_report_cells(self):
        self.assertEqual(subcells({'seed':0,'task':'composed'},'twohop'),[(0,'composed')])
        self.assertEqual(subcells({},'profile'),[(0,'counterfact_profile')])


if __name__=='__main__':unittest.main()
