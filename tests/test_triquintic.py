from pathlib import Path
import numpy as np
import unittest
from qinterp import TriquinticScalarInterpolator

class TriquinticScalarInterpolatorTest(unittest.TestCase):

    def setUp(self):
        # import a test field
        tests_dir = Path(__file__).parent
        data_path = tests_dir / "fields" / "Example3DScalarField.csv"
        field = np.genfromtxt(data_path, delimiter=',')

        # Create an instance of the interpolator
        self.interpolator = TriquinticScalarInterpolator(field)

    def test_field(self):
        # Test the field method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.field(point)
        self.assertTrue(np.isnan(result))  # Assert the result is NaN

    def test_gradient(self):
        # Test the gradient method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.gradient(point)
        expected_gradient = np.empty((3,))
        expected_gradient[:] = np.nan
        np.testing.assert_array_equal(result, expected_gradient)  # Assert the result is NaN

    def test_hessian(self):
        # Test the hessian method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.hessian(point)
        expected_hessian = np.empty((3, 3))
        expected_hessian[:] = np.nan
        np.testing.assert_array_equal(result, expected_hessian)  # Assert the result is NaN

if __name__ == '__main__':
    unittest.main()