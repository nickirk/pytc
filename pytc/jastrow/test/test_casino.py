import unittest
import numpy as np
import os
from pytc.jastrow import CASINO
from pytc.utils.parser import parse_casl

class TestCASINO(unittest.TestCase):
    """Tests for CASINO class."""

    def setUp(self):
        """Set up test fixtures."""
        # Load test parameters
        test_dir = os.path.dirname(os.path.abspath(__file__))
        casl_path = os.path.join(test_dir, '../../utils/test/parameters.casl')
        self.params = parse_casl(casl_path)
        self.casino = CASINO(self.params)

    def test_get_nuc_groups_no_mol(self):
        """Test nuclear grouping with no molecule."""
        with self.assertRaises(ValueError):
            self.casino._get_nuc_groups()

    def test_get_nuc_coords_no_mol(self):
        """Test getting nuclear coordinates with no molecule."""
        with self.assertRaises(ValueError):
            self.casino._get_nuc_coords()

    def test_compute_term_2e0n_with_params(self):
        """Test electron-electron term computation with real parameters."""
        # Create simple electron positions
        r1 = np.array([[0., 0., 0.], [1., 0., 0.]])
        r2 = np.array([[0.5, 0., 0.], [1.5, 0., 0.]])

        # Get the e-e term from params
        terms = self.params.get('JASTROW', {}).values()
        ee_term = next(term for term in terms if term['Rank'] == [2, 0])
        
        # Compute the term
        result = self.casino.compute_term_2e0n(ee_term, r1, r2)
        
        # Result should be finite and have expected shape
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(result)))
        
        # Test symmetry property with r1 == r2
        result_symm = self.casino.compute_term_2e0n(ee_term, r1, r1)
        np.testing.assert_array_almost_equal(result_symm, result_symm.T)

    def test_compute_term_1e1n_with_params(self):
        """Test electron-nuclear term computation with real parameters."""
        # Simple electron and nuclear positions
        r1 = np.array([[0., 0., 0.], [1., 0., 0.]])
        r_nuc = np.array([[0., 0., 0.5], [0., 0., -0.5]])
        
        # Mock mol object
        class MockMol:
            def atom_charges(self): return np.array([7.0, 7.0])
            def atom_coords(self): return r_nuc
        self.casino.mol = MockMol()
        
        # Mock nuclear groups
        nuc_groups = {'n1': ['n1', 'n2']}
        
        # Get the e-n term from params
        terms = self.params.get('JASTROW', {}).values()
        en_term = next(term for term in terms if term['Rank'] == [1, 1])
        
        # Compute the term
        result = self.casino.compute_term_1e1n(en_term, r1, r_nuc, nuc_groups)
        
        # Result should be finite
        self.assertTrue(np.all(np.isfinite(result)))

    def test_compute_e_n_distances(self):
        """Test electron-nuclear distance computation."""
        r1 = np.array([[0., 0., 0.], [1., 1., 1.]])
        r_nuc = np.array([[0., 0., 1.], [2., 0., 0.]])
        
        distances = self.casino._compute_e_n_distances(r1, r_nuc)
        
        expected = np.array([
            [1.0, 2.0],
            [np.sqrt(2), np.sqrt(3)]
        ])
        np.testing.assert_array_almost_equal(distances, expected)

    def test_full_jastrow_evaluation(self):
        """Test full Jastrow factor evaluation with all terms."""
        # Simple test configuration
        r1 = np.array([[0., 0., 0.], [1., 0., 0.]])
        r2 = np.array([[0.5, 0., 0.], [1.5, 0., 0.]])
        r_nuc = np.array([[0., 0., 0.5]])
        
        # Mock mol object
        class MockMol:
            def atom_charges(self): return np.array([7.0])
            def atom_coords(self): return r_nuc
        self.casino.mol = MockMol()
        
        nuc_groups = {'n1': ['n1']}
        
        # Evaluate full Jastrow factor
        result = self.casino(r1, r2, r_nuc, nuc_groups)
        
        # Result should be finite and real
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(result.shape, (2, 2))

    def test_orbital_cusp_correction(self):
        """Test orbital cusp correction computation."""
        # Get a term with orbital cusp correction
        terms = self.params.get('JASTROW', {}).values()
        cusp_term = next((term for term in terms 
                         if term.get('e-n cutoff', {}).get('Type') == 'orbital cusp'), None)
        
        if cusp_term is not None:
            r = np.array([0.1, 0.5, 1.0])
            result = self.casino.compute_orbital_cusp_correction(r, cusp_term)
            
            # Results should be finite
            self.assertTrue(np.all(np.isfinite(result)))
            
            # Test cutoff behavior
            r_large = np.array([1000.0])
            result_large = self.casino.compute_orbital_cusp_correction(r_large, cusp_term)
            np.testing.assert_array_almost_equal(result_large, 0.0)

    def test_spin_dependent_terms(self):
        """Test error handling for spin-dependent terms."""
        # Example term without '1=2' rule
        term = {'Rules': ['other_rule']}
        r1 = np.array([[0., 0., 0.]])
        r2 = np.array([[1., 1., 1.]])
        
        with self.assertRaises(NotImplementedError):
            self.casino.compute_term_2e0n(term, r1, r2)

if __name__ == "__main__":
    unittest.main()
