from pathlib import Path
import numpy as np
import unittest
from qinterp import TriquinticScalarInterpolator, TricubicScalarInterpolator

class TriquinticScalarInterpolatorTest(unittest.TestCase):

    def setUp(self):
        # import a test field
        # tests_dir = Path(__file__).parent
        # data_path = tests_dir / "fields" / "Example3DScalarField.csv"
        # field = np.genfromtxt(data_path, delimiter=',')
        length = 101
        x = np.linspace(-1,1,length)
        z = x
        y = x

        xxx, yyy, zzz = np.meshgrid(x,y,z, indexing='ij')

        positions = np.column_stack([xxx.ravel(), yyy.ravel(), zzz.ravel()])
        a_stacked = np.exp(-(positions[:,0]**2 + positions[:,1]**2 + positions[:,2]**2)/(2*0.1))

        field = np.zeros((length**3, 4))
        field[:,:3] = positions
        field[:,3] = a_stacked

        # Create an instance of the interpolator
        self.interpolator = TriquinticScalarInterpolator(field)

    def test_field(self):
        # Test the field method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.field(point)
        self.assertTrue(np.isnan(result))  # Assert the result is NaN
        point = np.array([0.96, 0, 0])
        result = self.interpolator.field(point)
        self.assertTrue(np.isnan(result))  # Assert the result is NaN
        self.assertTrue(self.interpolator.xIntMax == 0.96)  # Assert the xIntMin is -0.96

        x1 = np.linspace(-0.96,0.96,301)
        y1 = np.ones((301,))*0.0
        z1 = np.ones((301,))*0.0
        coords = np.zeros((301,3))
        coords[:,0] = x1
        coords[:,1] = y1
        coords[:,2] = z1

        field_int = self.interpolator.field(coords)
        print(field_int)
        self.assertTrue(np.allclose(np.exp(-(x1[:-1]**2)/(2*0.1)), field_int[:-1]))
        


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