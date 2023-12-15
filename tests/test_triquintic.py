from pathlib import Path
import numpy as np
import unittest
from qinterp import TriquinticScalarInterpolator, TricubicScalarInterpolator
import matplotlib.pyplot as plt

class TriquinticScalarInterpolatorTest(unittest.TestCase):

    def setUp(self):
        # import a test field
        # tests_dir = Path(__file__).parent
        # data_path = tests_dir / "fields" / "Example3DScalarField.csv"
        # field = np.genfromtxt(data_path, delimiter=',')
        self.length = 101
        x = np.linspace(-1,1,self.length)
        z = x
        y = x

        xxx, yyy, zzz = np.meshgrid(x,y,z, indexing='ij')

        positions = np.column_stack([xxx.ravel(), yyy.ravel(), zzz.ravel()])
        # a_stacked = np.exp(-(positions[:,0]**2 + positions[:,1]**2 + positions[:,2]**2)/(2*0.1))
        a_stacked = positions[:,0] ** 3 + positions[:,1] ** 3 + positions[:,2] ** 3

        self.field = np.zeros((self.length**3, 4))
        self.field[:,:3] = positions
        self.field[:,3] = a_stacked

        self.length1 = 901
        ratio = (self.length1 - 1) // (self.length - 1)
        edge1 = 1

        self.x1 = np.linspace(-edge1, edge1, self.length1)[2*ratio:-2*ratio]
        self.y1 = self.x1*0.0
        self.z1 = self.x1*0.0
        self.coords = np.zeros((len(self.x1),3))
        self.coords[:,0] = self.x1
        self.coords[:,1] = self.y1
        self.coords[:,2] = self.z1

        # Create an instance of the interpolator
        self.interpolator = TriquinticScalarInterpolator(self.field)

    def test_field(self):
        # Test the field method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.field(point)
        # self.assertTrue(np.isnan(result))  # Assert the result is NaN
        # point = np.array([0.96, 0, 0])
        point = np.array([0, 0, 0])
        result = self.interpolator.field(point)
        print(result)
        print(self.field[self.length**2*(self.length-1)//2 + self.length*(self.length-1)//2 + (self.length-1)//2,3])
        self.assertTrue(np.allclose(result, self.field[self.length**2*(self.length-1)//2 + self.length*(self.length-1)//2 + (self.length-1)//2, 3], rtol=1e-2))  # Assert the result is NaN
        # self.assertTrue(self.interpolator.xIntMax == 0.96)  # Assert the xIntMin is -0.96

        field_int = self.interpolator.field(self.coords)
        print(np.max(np.abs(field_int[:-1].T - np.exp(-(self.x1[:-1]**2)/(2*0.1)))))
        plt.plot(self.x1, field_int[:], 'o-', label='field_int')
        plt.plot(self.field[:self.length, 2], self.field[(self.length**2*(self.length-1)//2 + self.length*(self.length-1)//2):(self.length**2*(self.length-1)//2 + self.length*(self.length-1)//2 + self.length), 3:], 'o-', label='field_init')
        # plt.plot(self.x1, self.x1 ** 3, 'o-')
        # print(max(np.abs(field_int[:-1] - np.exp(-(x1[:-1]**2)/(2*0.1)))))
        # self.assertTrue(np.allclose(np.exp(-(x1[:-1]**2)/(2*0.1)), field_int[:-1]))
        


    def test_gradient(self):
        # Test the gradient method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.gradient(point)
        expected_gradient = np.empty((3,))
        expected_gradient[:] = np.nan
        np.testing.assert_array_equal(result, expected_gradient)  # Assert the result is NaN

        # Test the field method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.field(point)
        self.assertTrue(np.isnan(result))  # Assert the result is NaN
        point = np.array([0.96, 0, 0])
        result = self.interpolator.field(point)
        # self.assertTrue(np.isnan(result))  # Assert the result is NaN
        # self.assertTrue(self.interpolator.xIntMax == 0.96)  # Assert the xIntMin is -0.96

        field_int = self.interpolator.field(self.coords)
        field_int_grad = self.interpolator.gradient(self.coords)
        numpy_gradient = np.gradient(field_int[:,0], 2/(self.length1-1))
        plt.plot(self.x1, field_int_grad[:, 0], 'o-', label='field_int_grad')
        plt.plot(self.x1, numpy_gradient, 'o-', label='numpy_gradient')

    def test_hessian(self):
        # Test the hessian method with a single point
        point = np.array([1.0, 2.0, 3.0])
        result = self.interpolator.hessian(point)
        expected_hessian = np.empty((3, 3))
        expected_hessian[:] = np.nan
        np.testing.assert_array_equal(result, expected_hessian)  # Assert the result is NaN

        field_int_grad = self.interpolator.gradient(self.coords)
        field_int_hess = self.interpolator.hessian(self.coords)
        numpy_hessian = np.gradient(field_int_grad[:,0], 2/(self.length1-1))
        plt.plot(self.x1, field_int_hess[:, 0, 0], 'o-', label='field_int_hess')
        plt.plot(self.x1, numpy_hessian, 'o-', label='numpy_hessian')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    unittest.main()