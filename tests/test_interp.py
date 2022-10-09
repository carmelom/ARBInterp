#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

from pathlib import Path
import numpy as np
from qinterp import TricubicScalarInterpolator
from ARBInterp.ARBInterp import tricubic
from scipy.interpolate import RegularGridInterpolator

tests_dir = Path(__file__).parent
data_path = tests_dir / "fields" / "Example3DScalarField.csv"
interp_data_path = tests_dir / "fields" / "Example3DScalarFieldResults.npz"


def test_make_interp_data():
    data = np.genfromtxt(data_path, delimiter=',')
    Run = tricubic(data)

    coords = np.zeros((20, 3))
    coords[:, 0] = np.linspace(-2e-3, 2e-3, 20)
    coords[:, 1] = np.linspace(-2e-3, 2e-3, 20)
    coords[:, 2] = np.linspace(-2e-3, 2e-3, 20)

    field, grad = Run.Query((coords[3]))
    fields, grads = Run.Query(coords)
    np.savez(interp_data_path,
             coords=coords, single_field=field, single_grad=grad,
             multi_field=fields, multi_grad=grads
             )
    assert interp_data_path.exists()


def test_scalar():
    """test that the new interpolator gives the same results
    as the original ARB one.
    The interpolated field is the one from examples/3D
    """
    data = np.genfromtxt(data_path, delimiter=',')
    tri = TricubicScalarInterpolator(data)

    A = np.load(interp_data_path)
    coords = A['coords']
    xi = coords[3]

    field = tri.field(xi)
    grad = tri.gradient(xi)

    assert np.allclose(field, A['single_field'])
    assert np.allclose(grad, A['single_grad'])

    fields = tri.field(coords)
    grads = tri.gradient(coords)

    assert np.allclose(fields, A['multi_field'])
    assert np.allclose(grads, A['multi_grad'])
    assert np.allclose(tri(coords[3], d=2), tri.hessian(coords)[3])


def test_scipy():
    data = np.genfromtxt(data_path, delimiter=',')
    tri = TricubicScalarInterpolator(data)
    points = tri.x, tri.y, tri.z
    values = tri.field_data

    grid_interp = RegularGridInterpolator(points, values, method="cubic")

    A = np.load(interp_data_path)
    coords = A['coords']

    field = grid_interp(coords)
    max_rtol = (abs(field.squeeze() - A['multi_field'].squeeze()) / abs(field.squeeze())).max()
    print(f"rtol: {max_rtol:.3e}")
    assert max_rtol < 2e-1
