#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

from pathlib import Path
import numpy as np
from qinterp import TricubicScalarInterpolator
from ARBinterp import tricubic as ARBTricubicInterpolator
from scipy.interpolate import RegularGridInterpolator

tests_dir = Path(__file__).parent
data_path = tests_dir / "fields" / "Example3DScalarField.csv"
interp_data_path = tests_dir / "fields" / "Example3DScalarFieldResults",


def make_interp_data():
    data = np.genfromtxt(data_path, delimiter=',')
    Run = ArbTricubicInterpolator(data)

    coords = np.zeros((20, 3))
    coords[:, 0] = np.linspace(-2e-3, 2e-3, 20)
    coords[:, 1] = np.linspace(-2e-3, 2e-3, 20)
    coords[:, 2] = np.linspace(-2e-3, 2e-3, 20)

    field, grad = Run.Query((coords[3]))
    fields, grads = Run.Query(coords)
    np.savez(
        coords=coords, single_field=field, single_grad=grad,
        multi_field=fields, multi_grad=grads
    )


def test_scalar():
    """test that the new interpolator gives the same results
    as the original ARB one.
    The interpolated field is the one from examples/3D
    """
    data_path = tests_dir / "fields" / "Example3DScalarField.csv"
    data = np.genfromtxt(data_path, delimiter=',')

    tri = TricubicScalarInterpolator(data)

    A = np.load(tests_dir / "fields" / "Example3DScalarFieldResults.npz")
    coords = A['coords']

    field, grad, hessian = tri.field1(coords[3], derivatives=True)

    assert np.allclose(field, A['single_field'])
    assert np.allclose(grad, A['single_grad'])

    fields = tri.field(coords)
    grads = tri.gradient(coords)

    assert np.allclose(fields, A['multi_field'])
    assert np.allclose(grads, A['multi_grad'])


def _test_scipy():
    data_path = tests_dir / "fields" / "Example3DScalarField.csv"
    data = np.genfromtxt(data_path, delimiter=',')

    tri = TricubicScalarInterpolator(data)
    points = tri.x, tri.y, tri.z
    values = tri.field_data

    grid_interp = RegularGridInterpolator(points, values, method="cubic")

    A = np.load(tests_dir / "fields" / "Example3DScalarFieldResults.npz")
    coords = A['coords']

    field = grid_interp(coords[3])

    assert np.allclose(field, A['single_field'])
