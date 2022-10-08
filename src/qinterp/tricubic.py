#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

import numpy as np
from .base import TricubicInterpolatorBase


class TricubicScalarInterpolator(TricubicInterpolatorBase):

    def __init__(self, field, *args, **kwargs):
        super().__init__(field, *args, **kwargs)
        self.alphan = np.zeros((64, self.nc + 1))
        self.alphan[:, -1] = np.nan
        self.Bn = self.inputfield[:, 3]
        # precalculate all coefficients
        # self.allCoeffs()

        self._call_single = self.field1
        self._call_array = self.field

    def calcCoefficients(self, alphaindex):
        # Find interpolation coefficients for a cuboid
        realindex = self.basePointInds[alphaindex]
        # Find other vertices of current cuboid, and all neighbours in 3x3x3 neighbouring array
        inds = self.neighbourInd(realindex)
        # Alpha coefficients
        self.alphan[:, alphaindex] = np.dot(self.A, self.Bn[inds])
        self.alphamask[alphaindex] = 1

    def field1(self, point, derivatives=False):
        """ Calculate field and derivatives at one single point
        """
        # Removes particles that are outside of interpolation volume
        if self.check_out_of_bounds(point):
            return np.nan

        queryIndex, (cuboidx, cuboidy, cuboidz) = self._cuboid_coordinates1(point)
        # interpolated values
        tn = self.alphan[:, queryIndex]

        # 4-vectors for finding interpolated values
        x = np.tile(np.array([1, cuboidx, cuboidx**2, cuboidx**3]), 16)
        y = np.tile(np.repeat(np.array([1, cuboidy, cuboidy**2, cuboidy**3]), 4), 4)
        z = np.repeat(np.array([1, cuboidz, cuboidz**2, cuboidz**3]), 16)

        # Magnitude
        norm = np.inner(tn, (x * y * z))
        if not derivatives:
            return norm

        # 4-vectors for finding interpolated gradients
        xx = np.tile(np.array([0, 1, 2 * cuboidx, 3 * cuboidx**2]), 16)
        yy = np.tile(np.repeat(np.array([0, 1, 2 * cuboidy, 3 * cuboidy**2]), 4), 4)
        zz = np.repeat(np.array([0, 1, 2 * cuboidz, 3 * cuboidz**2]), 16)

        # 4-vectors for finding interpolated hessian
        xxx = np.tile(np.array([0, 0, 2, 6 * cuboidx]), 16)
        yyy = np.tile(np.repeat(np.array([0, 0, 2, 6 * cuboidy]), 4), 4)
        zzz = np.repeat(np.array([0, 0, 2, 6 * cuboidz]), 16)

        # gradient, hessian
        grad = np.array([np.dot(tn, xx * y * z) / self.hx, np.dot(tn, x * yy * z) / self.hy, np.dot(tn, x * y * zz) / self.hz])
        hxx, hxy, hxz = np.dot(tn, xxx * y * z) / self.hx / self.hx, np.dot(tn, xx * yy * z) / self.hy / self.hx, np.dot(tn, xx * y * zz) / self.hz / self.hx
        hyy, hyz = np.dot(tn, x * yyy * z) / self.hy / self.hy, np.dot(tn, x * yy * zz) / self.hz / self.hy
        hzz = np.dot(tn, x * y * zzz) / self.hz / self.hz
        hess = np.array([
            [hxx, hxy, hxz],
            [hxy, hyy, hyz],
            [hxz, hyz, hzz],
        ])
        return norm, grad, hess

    def _cuboid_coordinates1(self, point):
        # How many cuboids in is query point
        iu = (point[0] - self.xIntMin) / self.hx
        iv = (point[1] - self.yIntMin) / self.hy
        iw = (point[2] - self.zIntMin) / self.hz
        # Finds base coordinates of cuboid particle is in
        ix = np.floor(iu)
        iy = np.floor(iv)
        iz = np.floor(iw)
        # particle coordinates in unit cuboid
        cuboidx = iu - ix
        cuboidy = iv - iy
        cuboidz = iw - iz
        # Returns index of base cuboid
        queryIndex = ix + iy * (self.nPos[0]) + iz * (self.nPos[0]) * (self.nPos[1])
        queryIndex = queryIndex.astype(int)

        queryCoords = np.asarray((iu - ix, iv - iy, iw - iz))

        # Calculate alpha for cuboid if it doesn't exist
        if self.alphamask[queryIndex] == 0:
            self.calcCoefficients(queryIndex)

        return queryIndex, queryCoords

    def field(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, cx, cy, cz = self._cuboid_coordinates(points)

        x, y, z = self._pack_coords(cx, cy, cz, d=0)
        # Return coefficient matrix values, give NaN for invalid locations
        tn = self.alphan[:, queryInds]
        tn = np.transpose(tn)

        field = np.reshape((tn * (x * y * z)).sum(axis=1), (-1, 1))
        return field

    def gradient(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, cx, cy, cz = self._cuboid_coordinates(points)

        x, y, z = self._pack_coords(cx, cy, cz, d=0)
        xx, yy, zz = self._pack_coords(cx, cy, cz, d=1)

        # Return coefficient matrix values, give NaN for invalid locations
        tn = self.alphan[:, queryInds]
        tn = np.transpose(tn)
        grad = np.transpose(np.array([((tn * (xx * y * z)) / self.hx).sum(axis=1), ((tn * (x * yy * z)) / self.hy).sum(axis=1), ((tn * (x * y * zz)) / self.hz).sum(axis=1)]))
        return grad

    def hessian(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, cx, cy, cz = self._cuboid_coordinates(points)

        x, y, z = self._pack_coords(cx, cy, cz, d=0)
        xx, yy, zz = self._pack_coords(cx, cy, cz, d=1)
        xxx, yyy, zzz = self._pack_coords(cx, cy, cz, d=2)

        # Return coefficient matrix values, give NaN for invalid locations
        tn = self.alphan[:, queryInds]
        tn = np.transpose(tn)

        hxx, hxy, hxz = ((tn * (xxx * y * z)) / self.hx / self.hx).sum(axis=1), ((tn * (xx * yy * z)) / self.hy / self.hx).sum(axis=1), ((tn * (xx * y * zz)) / self.hz / self.hx).sum(axis=1)
        hyy, hyz = ((tn * (x * yyy * z)) / self.hy / self.hy).sum(axis=1), ((tn * (x * yy * zz)) / self.hz / self.hy).sum(axis=1)
        hzz = ((tn * (x * y * zzz)) / self.hz / self.hz).sum(axis=1)

        hess = np.transpose(np.array([
            [hxx, hxy, hxz],
            [hxy, hyy, hyz],
            [hxz, hyz, hzz],
        ]))

        return hess

    def _cuboid_coordinates(self, points):
        # Coords in cuboids
        iu = (points[:, 0] - self.xIntMin) / self.hx
        iv = (points[:, 1] - self.yIntMin) / self.hy
        iw = (points[:, 2] - self.zIntMin) / self.hz

        ### Finds base coordinates of cuboid particles are in ###
        ix = np.floor(iu)
        iy = np.floor(iv)
        iz = np.floor(iw)

        ### Returns indices of base cuboids ###
        queryInds = ix + iy * (self.nPos[0]) + iz * (self.nPos[0]) * (self.nPos[1])
        queryInds[np.where(np.isnan(queryInds))] = self.nc
        queryInds = queryInds.astype(int)

        # Coordinates of the sample in unit cuboid
        cx, cy, cz = iu - ix, iv - iy, iw - iz

        # Returns the interpolate magnitude and / or gradients at the query coordinates
        if len(queryInds[np.where(self.alphamask[queryInds] == 0)[0]]) > 0:
            list(map(self.calcCoefficients, queryInds[np.where(self.alphamask[queryInds] == 0)[0]]))

        return queryInds, cx, cy, cz

    def _pack_coords(self, cx, cy, cz, d=0):
        zero = np.zeros(len(cx))
        one = np.ones(len(cx))
        if d == 0:
            x = np.tile(np.transpose(np.array([one, cx, cx**2, cx**3])), 16)
            y = np.tile(np.repeat(np.transpose(np.array([one, cy, cy**2, cy**3])), 4, axis=1), 4)
            z = np.repeat(np.transpose(np.array([one, cz, cz**2, cz**3])), 16, axis=1)
        elif d == 1:
            # Derivatives
            x = np.tile(np.transpose(np.array([zero, one, 2 * cx, 3 * cx**2])), 16)
            y = np.tile(np.repeat(np.transpose(np.array([zero, one, 2 * cy, 3 * cy**2])), 4, axis=1), 4)
            z = np.repeat(np.transpose(np.array([zero, one, 2 * cz, 3 * cz**2])), 16, axis=1)
        elif d == 2:
            x = np.tile(np.transpose(np.array([zero, zero, 2 * one, 6 * cx])), 16)
            y = np.tile(np.repeat(np.transpose(np.array([zero, zero, 2 * one, 6 * cy])), 4, axis=1), 4)
            z = np.repeat(np.transpose(np.array([zero, zero, 2 * one, 6 * cz])), 16, axis=1)
        else:
            raise NotImplementedError
        return x, y, z


class TricubicVectorInterpolator(TricubicInterpolatorBase):

    def __init__(self, field, *args, **kwargs):
        raise NotImplementedError
