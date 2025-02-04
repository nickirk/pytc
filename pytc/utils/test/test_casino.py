import unittest
import os

class TestCasinoParameters(unittest.TestCase):
    def setUp(self):
        self.param_file = os.path.join(os.path.dirname(__file__), 'parameters.casl')
    
    def test_parse_casl(self):
        from pytc.utils.casino import parse_casl
        params = parse_casl(self.param_file)
        
        # Test basic structure
        self.assertIn("JASTROW", params)
        self.assertEqual(len(params["JASTROW"].keys()), 6)  # 6 terms
        
        # Test Term 1
        term1 = params["JASTROW"]["TERM 1"]
        self.assertEqual(term1["Rank"], [2, 0])
        self.assertEqual(term1["e-e basis"]["Type"], "natural power")
        self.assertEqual(term1["e-e basis"]["Order"], 9)
        
        # Test some parameters
        self.assertEqual(term1["Linear parameters"]["Channel 1-2"]["c_2"][0], 0.17116191470246386)
        self.assertEqual(term1["Linear parameters"]["Channel 1-2"]["c_2"][1], "optimizable")

    def test_parse_casl_invalid_file(self):
        from pytc.utils.casino import parse_casl
        
        with self.assertRaises(FileNotFoundError):
            parse_casl("nonexistent.casl")

    def test_parse_casl_term5(self):
        from pytc.utils.casino import parse_casl
        params = parse_casl(self.param_file)
        
        term5 = params["JASTROW"]["TERM 5"]
        # Check basic structure
        self.assertEqual(term5["Rank"], [1, 1])
        self.assertEqual(term5["e-n cusp"], "T")
        self.assertEqual(term5["Rules"], ["1=2", "Z", "!N2", "!N3", "!N4"])
        self.assertEqual(term5["e-n basis"]["Type"], "none")
        
        # Check orbital cusp parameters
        cusp = term5["e-n cutoff"]
        self.assertEqual(cusp["Type"], "orbital cusp")
        self.assertEqual(cusp["Constants"]["norb"], 1)
        self.assertEqual(cusp["Constants"]["ngrid"], 999)
        self.assertEqual(cusp["Constants"]["Z"], 7.0)
        self.assertEqual(cusp["Constants"]["C"], 0.0)
        self.assertEqual(cusp["Constants"]["L_grid"], 0.17857142857142855)
        
        # Check some orbital values
        orbital = cusp["Constants"]["Orbital 1"]
        self.assertEqual(orbital["phi_0"], 6.0094243060681904)
        self.assertEqual(orbital["phi_1"], 6.0093041288121842)
        self.assertEqual(orbital["phi_999"], 1.8105994855870560)

    def test_parse_casl_term6(self):
        from pytc.utils.casino import parse_casl
        params = parse_casl(self.param_file)
        
        term6 = params["JASTROW"]["TERM 6"]
        # Check basic structure
        self.assertEqual(term6["Rank"], [1, 1])
        self.assertEqual(term6["e-n cusp"], "T")
        self.assertEqual(term6["Rules"], ["1=2", "Z", "!N1"])
        self.assertEqual(term6["e-n basis"]["Type"], "none")
        
        # Check orbital cusp parameters
        cusp = term6["e-n cutoff"]
        self.assertEqual(cusp["Type"], "orbital cusp")
        self.assertEqual(cusp["Constants"]["norb"], 1)
        self.assertEqual(cusp["Constants"]["ngrid"], 999)
        self.assertEqual(cusp["Constants"]["Z"], 1.0)
        self.assertEqual(cusp["Constants"]["C"], 0.0)
        self.assertEqual(cusp["Constants"]["L_grid"], 1.2500000000000000)
        
        # Check some orbital values
        orbital = cusp["Constants"]["Orbital 1"]
        self.assertEqual(orbital["phi_0"], 0.32923116737332220)
        self.assertEqual(orbital["phi_1"], 0.32922879594635573)
        self.assertEqual(orbital["phi_999"], 0.024721450602844065)

if __name__ == '__main__':
    unittest.main()