#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 02/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

'''
Module docstring
'''

import numpy as np


class TricubicInterpolator:
    """Factory class
    """

    def __new__(cls, field, *args, **kwargs):
        if field.ndim == 2 and field.shape[1] == 4:
            return TricubicScalarInterpolator(field)
        elif field.ndim == 2 and field.shape[1] == 6:
            return TricubicVectorInterpolator(field)
        else:
            sys.exit('--- Input not shaped as expected - should be N x 4 or N x 6 ---')


class TricubicInterpolatorBase:

    def __init__(self, field, *args, **kwargs):
        self.eps = 10 * np.finfo(float).eps  # Machine precision of floating point number
        # Load field passed to class - can be x,y,z or norm of vector field, or scalar
        self.inputfield = field
        # Analyse field, get shapes etc
        self.getFieldParams()
        # Make coefficient matrix
        self.A = self.makeAMatrix()

        # Mask to identify where coefficients exist
        self.alphamask = np.zeros((self.nc + 1, 1))
        self.alphamask[-1] = 1

    def _call(self, query):
        try:
            if query.shape[1] > 1:
                return self._call_array(query)
            else:
                return self._call_single(query)
        except IndexError:
            return self._call_single(query)

    def __call__(self, query):
        return self._call(query)

    def Query(self, query):
        # for backward copatiblility
        return self._call(query)

    def _call_single(self, point):
        raise NotImplementedError

    def _call_array(self, point):
        raise NotImplementedError

    def getFieldParams(self):
        # Make sure coords are sorted correctly
        # This sorts the data in 'F' style
        self.inputfield = self.inputfield[self.inputfield[:, 0].argsort()]
        self.inputfield = self.inputfield[self.inputfield[:, 1].argsort(kind='mergesort')]
        self.inputfield = self.inputfield[self.inputfield[:, 2].argsort(kind='mergesort')]

        # Analyse field
        xzero = self.inputfield[np.where(self.inputfield[:, 0] == self.inputfield[0, 0])[0]]
        yzero = self.inputfield[np.where(self.inputfield[:, 1] == self.inputfield[0, 1])[0]]
        xaxis = yzero[np.where(yzero[:, 2] == yzero[0, 2])[0]]
        yaxis = xzero[np.where(xzero[:, 2] == xzero[0, 2])[0]]
        zaxis = xzero[np.where(xzero[:, 1] == xzero[0, 1])[0]]

        nPosx = len(xaxis)  # These give the length of the interpolation volume along each axis
        nPosy = len(yaxis)
        nPosz = len(zaxis)

        self.nPos = np.array([nPosx - 3, nPosy - 3, nPosz - 3])  # number of interpolatable cuboids per axis

        self.x = xaxis[:, 0]  # axis coordinates
        self.y = yaxis[:, 1]
        self.z = zaxis[:, 2]
        if self.inputfield.shape[1] == 4:
            self.field_data = self.inputfield[:, 3].reshape(nPosx, nPosy, nPosz, order='F')
        else:
            self.field_data = np.stack([self.inputfield[:, j].reshape(nPosx, nPosy, nPosz, order='F')
                                        for j in range(3, 6)], axis=0)

        self.hx = np.abs((xaxis[0, 0] - xaxis[1, 0]))  # grid spacing along each axis
        self.hy = np.abs((yaxis[0, 1] - yaxis[1, 1]))
        self.hz = np.abs((zaxis[0, 2] - zaxis[1, 2]))

        self.xIntMin = self.inputfield[1, 0]  # Minimal value of x that can be interpolated
        self.yIntMin = self.inputfield[nPosx, 1]
        self.zIntMin = self.inputfield[nPosx * nPosy, 2]
        self.xIntMax = self.inputfield[-2, 0]  # Maximal value of x that can be interpolated
        self.yIntMax = self.inputfield[-2 * nPosx, 1]
        self.zIntMax = self.inputfield[-2 * nPosx * nPosy, 2]

        # Find base indices of all interpolatable cuboids
        minI = nPosx * nPosy + nPosx + 1
        self.basePointInds = minI + np.arange(0, nPosx - 3, 1)
        temp = np.array([self.basePointInds + i * nPosx for i in range(nPosy - 3)])
        self.basePointInds = np.reshape(temp, (1, len(temp) * len(temp[0])))[0]
        temp = np.array([self.basePointInds + i * nPosx * nPosy for i in range(nPosz - 3)])
        self.basePointInds = np.reshape(temp, (1, len(temp) * len(temp[0])))[0]
        self.basePointInds = np.sort(self.basePointInds)

        # Number of interpolatable cuboids
        self.nc = len(self.basePointInds)

    def makeAMatrix(self):
        # Creates tricubic interpolation matrix and finite difference matrix and combines them
        # Interpolation matrix
        corners = np.array(([[i, j, k] for k in range(2) for j in range(2) for i in range(2)])).astype(float).T
        exp = [[i, j, k] for k in range(4) for j in range(4) for i in range(4)]
        B = np.zeros((64, 64), dtype=np.float64)

        for i in range(64):
            ex, ey, ez = exp[i][0], exp[i][1], exp[i][2]    #
            for k in range(8):
                x, y, z = corners[0, k], corners[1, k], corners[2, k]
                B[0 * 8 + k, i] = x**ex * y**ey * z**ez
                B[1 * 8 + k, i] = ex * x**(abs(ex - 1)) * y**ey * z**ez
                B[2 * 8 + k, i] = x**ex * ey * y**(abs(ey - 1)) * z**ez
                B[3 * 8 + k, i] = x**ex * y**ey * ez * z**(abs(ez - 1))
                B[4 * 8 + k, i] = ex * x**(abs(ex - 1)) * ey * y**(abs(ey - 1)) * z**ez
                B[5 * 8 + k, i] = ex * x**(abs(ex - 1)) * y**ey * ez * z**(abs(ez - 1))
                B[6 * 8 + k, i] = x**ex * ey * y**(abs(ey - 1)) * ez * z**(abs(ez - 1))
                B[7 * 8 + k, i] = ex * x**(abs(ex - 1)) * ey * y**(abs(ey - 1)) * ez * z**(abs(ez - 1))

        # This makes a finite-difference matrix to return the components of the "b"-vector
        # needed in the alpha calculation, in "cuboid" coordinates
        C = np.array((21, 22, 25, 26, 37, 38, 41, 42))
        D = np.zeros((64, 64))

        for i in range(8):
            D[i, C[i]] = 1

        for i, j in enumerate(range(8, 16, 1)):
            D[j, C[i] - 1] = -0.5
            D[j, C[i] + 1] = 0.5

        for i, j in enumerate(range(16, 24, 1)):
            D[j, C[i] - 4] = -0.5
            D[j, C[i] + 4] = 0.5

        for i, j in enumerate(range(24, 32, 1)):
            D[j, C[i] - 16] = -0.5
            D[j, C[i] + 16] = 0.5

        for i, j in enumerate(range(32, 40, 1)):
            D[j, C[i] + 5] = 0.25
            D[j, C[i] - 3] = -0.25
            D[j, C[i] + 3] = -0.25
            D[j, C[i] - 5] = 0.25

        for i, j in enumerate(range(40, 48, 1)):
            D[j, C[i] + 17] = 0.25
            D[j, C[i] - 15] = -0.25
            D[j, C[i] + 15] = -0.25
            D[j, C[i] - 17] = 0.25

        for i, j in enumerate(range(48, 56, 1)):
            D[j, C[i] + 20] = 0.25
            D[j, C[i] - 12] = -0.25
            D[j, C[i] + 12] = -0.25
            D[j, C[i] - 20] = 0.25

        for i, j in enumerate(range(56, 64, 1)):
            D[j, C[i] + 21] = 0.125
            D[j, C[i] + 13] = -0.125
            D[j, C[i] + 19] = -0.125
            D[j, C[i] + 11] = 0.125
            D[j, C[i] - 11] = -0.125
            D[j, C[i] - 19] = 0.125
            D[j, C[i] - 13] = 0.125
            D[j, C[i] - 21] = -0.125

        A = np.matmul(np.linalg.inv(B), D)
        return A

    def neighbourInd(self, ind0):
        # For base index ind0 this finds all 64 vertices of the 3x3x3 range of cuboids around it
        # It also returns the 7 neighbouring points
        newind0 = ind0 - 1 - (self.nPos[0] + 3) * (self.nPos[1] + 4)
        bInds = np.zeros(64)
        bInds[0] = newind0
        bInds[1] = bInds[0] + 1
        bInds[2] = bInds[1] + 1
        bInds[3] = bInds[2] + 1
        bInds[4:8] = bInds[:4] + self.nPos[0] + 3
        bInds[8:12] = bInds[4:8] + self.nPos[0] + 3
        bInds[12:16] = bInds[8:12] + self.nPos[0] + 3
        bInds[16:32] = bInds[:16] + (self.nPos[0] + 3) * (self.nPos[1] + 3)
        bInds[32:48] = bInds[16:32] + (self.nPos[0] + 3) * (self.nPos[1] + 3)
        bInds[48:] = bInds[32:48] + (self.nPos[0] + 3) * (self.nPos[1] + 3)
        bInds = bInds.astype(int)
        return bInds

    def allCoeffs(self):
        """Precalculate all cuboid coefficients
        """
        allinds = np.arange(self.nc)
        list(map(self.calcCoefficients, allinds))

    def check_out_of_bounds(self, query):
        """ Checks if query point is within interpolation bounds
            True if within
        """
        return query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax

    def nan_out_of_bounds(self, query):
        # Removes particles that are outside of interpolation volume
        query[np.where(query[:, 0] < self.xIntMin)[0]] = np.nan
        query[np.where(query[:, 0] > self.xIntMax)[0]] = np.nan
        query[np.where(query[:, 1] < self.yIntMin)[0]] = np.nan
        query[np.where(query[:, 1] > self.yIntMax)[0]] = np.nan
        query[np.where(query[:, 2] < self.zIntMin)[0]] = np.nan
        query[np.where(query[:, 2] > self.zIntMax)[0]] = np.nan
        return query


class TricubicScalarInterpolator(TricubicInterpolatorBase):

    def __init__(self, field, *args, **kwargs):
        super().__init__(field, *args, **kwargs)
        self.alphan = np.zeros((64, self.nc + 1))
        self.alphan[:, -1] = np.nan
        self.Bn = self.inputfield[:, 3]
        # precalculate all coefficients
        print("Calculate all coefficients")
        self.allCoeffs()

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

        return queryIndex, queryCoords

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
        queryCoords = np.stack((iu - ix, iv - iy, iw - iz), axis=1)
        return queryInds, queryCoords

    def field(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, queryCoords = self._cuboid_coordinates(points)
        cx, cy, cz = queryCoords.T
        N = len(points)

        x, y, z = self._pack_coords(cx, cy, cz, d=0)
        # Return coefficient matrix values, give NaN for invalid locations
        tn = self.alphan[:, queryInds]
        tn = np.transpose(tn)

        field = np.reshape((tn * (x * y * z)).sum(axis=1), (N, 1))
        return field

    def gradient(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, queryCoords = self._cuboid_coordinates(points)
        cx, cy, cz = queryCoords.T
        N = len(points)

        x, y, z = self._pack_coords(cx, cy, cz, d=0)
        xx, yy, zz = self._pack_coords(cx, cy, cz, d=1)

        # Return coefficient matrix values, give NaN for invalid locations
        tn = self.alphan[:, queryInds]
        tn = np.transpose(tn)
        grad = np.transpose(np.array([((tn * (xx * y * z)) / self.hx).sum(axis=1), ((tn * (x * yy * z)) / self.hy).sum(axis=1), ((tn * (x * y * zz)) / self.hz).sum(axis=1)]))
        return grad

    def hessian(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, queryCoords = self._cuboid_coordinates(points)
        cx, cy, cz = queryCoords.T
        N = len(points)

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


"""
class TricubicVectorInterpolator(TricubicInterpolatorBase):

	def __init__(self, field, *args, **kwargs):
		print('--- Scalar field, ignoring switches, interpolating for magnitude and gradient --- \n')

        # Determining which mode to run in
		if self.inputfield.shape[1] == 4:
            if not 'quiet' in args:
                print('--- Scalar field, ignoring switches, interpolating for magnitude and gradient --- \n')
            self.Query = self.Query2
            self.sQuery = self.sQuery2
            self._call_array = self._call_array2
            self.calcCoefficients = self.calcCoefficients2
            
        elif self.inputfield.shape[1] == 6:
            if 'mode' in kwargs:
                if kwargs['mode'] == 'vector':
                    if not 'quiet' in args:
                        print('--- Vector field, interpolating for vector components --- \n')
                    self.Query = self.Query1
                    self.sQuery = self.sQuery1
                    self._call_array = self._call_array1
                    self.calcCoefficients = self.calcCoefficients1
                    self.alphax = np.zeros((64, self.nc + 1 + 1))
                    self.alphay = np.zeros((64, self.nc + 1 + 1))
                    self.alphaz = np.zeros((64, self.nc + 1 + 1))
                    self.alphax[:, -1] = self.alphay[:, -1] = self.alphaz[:, -1] = np.nan
                    self.Bx = self.inputfield[:, 3]
                    self.By = self.inputfield[:, 4]
                    self.Bz = self.inputfield[:, 5]
                elif kwargs['mode'] == 'norm':
                    if not 'quiet' in args:
                        print('--- Vector field, interpolating for magnitude and gradient --- \n')
                    self.Query = self.Query2
                    self.sQuery = self.sQuery2
                    self._call_array = self._call_array2
                    self.calcCoefficients = self.calcCoefficients2
                    self.alphan = np.zeros((64, self.nc + 1))
                    self.alphan[:, -1] = np.nan
                    self.Bn = np.linalg.norm(self.inputfield[:, 3:], axis=1)
                elif kwargs['mode'] == 'both':
                    if not 'quiet' in args:
                        print('--- Vector field, interpolating vector components plus magnitude and gradient --- \n')
                    self.Query = self.Query3
                    self.sQuery = self.sQuery3
                    self._call_array = self._call_array3
                    self.calcCoefficients = self.calcCoefficients3
                    self.alphax = np.zeros((64, self.nc + 1))
                    self.alphay = np.zeros((64, self.nc + 1))
                    self.alphaz = np.zeros((64, self.nc + 1))
                    self.alphan = np.zeros((64, self.nc + 1))
                    self.alphax[:, -1] = self.alphay[:, -1] = self.alphaz[:, -1] = self.alphan[:, -1] = np.nan
                    self.Bx = self.inputfield[:, 3]
                    self.By = self.inputfield[:, 4]
                    self.Bz = self.inputfield[:, 5]
                    self.Bn = np.linalg.norm(self.inputfield[:, 3:], axis=1)
                else:
                    if not 'quiet' in args:
                        print('--- Vector field, invalid option, defaulting to interpolating for vector components --- \n')
                    self.Query = self.Query1
                    self.sQuery = self.sQuery1
                    self._call_array = self._call_array1
                    self.calcCoefficients = self.calcCoefficients1
                    self.alphax = np.zeros((64, self.nc + 1))
                    self.alphay = np.zeros((64, self.nc + 1))
                    self.alphaz = np.zeros((64, self.nc + 1))
                    self.alphax[:, -1] = self.alphay[:, -1] = self.alphaz[:, -1] = np.nan
                    self.Bx = self.inputfield[:, 3]
                    self.By = self.inputfield[:, 4]
                    self.Bz = self.inputfield[:, 5]
            else:
                if not 'quiet' in args:
                    print('--- Vector field, no option selected, defaulting to interpolating for vector components --- \n')
                self.Query = self.Query1
                self.sQuery = self.sQuery1
                self._call_array = self._call_array1
                self.calcCoefficients = self.calcCoefficients1
                self.alphax = np.zeros((64, self.nc + 1))
                self.alphay = np.zeros((64, self.nc + 1))
                self.alphaz = np.zeros((64, self.nc + 1))
                self.alphax[:, -1] = self.alphay[:, -1] = self.alphaz[:, -1] = np.nan
                self.Bx = self.inputfield[:, 3]
                self.By = self.inputfield[:, 4]
                self.Bz = self.inputfield[:, 5]
        else:
            sys.exit('--- Input not shaped as expected - should be N x 4 or N x 6 ---')

    def Query1(self, query):
        try:
            if query.shape[1] > 1:
                comps = self._call_array1(query)
                return comps
            else:
                comps = self.sQuery1(query)
                return comps
        except IndexError:
            comps = self.sQuery1(query)
            return comps


    def Query3(self, query):
        try:
            if query.shape[1] > 1:
                comps, norms, grads = self._call_array3(query)
                return comps, norms, grads
            else:
                comps, norm, grad = self.sQuery3(query)
                return comps, norm, grad
        except IndexError:
            comps, norm, grad = self.sQuery3(query)
            return comps, norm, grad

    def sQuery1(self, query):
        # Removes particles that are outside of interpolation volume
        if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax:
            return np.nan
        else:
            # How many cuboids in is query point
            iu = (query[0] - self.xIntMin) / self.hx
            iv = (query[1] - self.yIntMin) / self.hy
            iw = (query[2] - self.zIntMin) / self.hz
            # Finds base coordinates of cuboid particle is in
            ix = np.floor(iu)
            iy = np.floor(iv)
            iz = np.floor(iw)
            # particle coordinates in unit cuboid
            cuboidx = iu - ix
            cuboidy = iv - iy
            cuboidz = iw - iz
            # Returns index of base cuboid
            self.queryInd = ix + iy * (self.nPos[0]) + iz * (self.nPos[0]) * (self.nPos[1])
            self.queryInd = self.queryInd.astype(int)
            # Calculate alpha for cuboid if it doesn't exist
            if self.alphamask[self.queryInd] == 0:
                self.calcCoefficients(self.queryInd)
            # 4-vectors for finding interpolated values
            xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
            yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
            zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
            # 4-vector summation components
            x = np.tile(xvec, 16)
            y = np.tile(np.repeat(yvec, 4), 4)
            z = np.repeat(zvec, 16)
            # interpolated values
            compx = np.inner(self.alphax[:, self.queryInd], (x * y * z))
            compy = np.inner(self.alphay[:, self.queryInd], (x * y * z))
            compz = np.inner(self.alphaz[:, self.queryInd], (x * y * z))
            # Components
            return np.array((compx, compy, compz))


    def sQuery3(self, query):
        # Removes particles that are outside of interpolation volume
        if query[0] < self.xIntMin or query[0] > self.xIntMax or query[1] < self.yIntMin or query[1] > self.yIntMax or query[2] < self.zIntMin or query[2] > self.zIntMax:
            return np.nan
        else:
            # How many cuboids in is query point
            iu = (query[0] - self.xIntMin) / self.hx
            iv = (query[1] - self.yIntMin) / self.hy
            iw = (query[2] - self.zIntMin) / self.hz
            # Finds base coordinates of cuboid particle is in
            ix = np.floor(iu)
            iy = np.floor(iv)
            iz = np.floor(iw)
            # particle coordinates in unit cuboid
            cuboidx = iu - ix
            cuboidy = iv - iy
            cuboidz = iw - iz
            # Returns index of base cuboid
            self.queryInd = ix + iy * (self.nPos[0]) + iz * (self.nPos[0]) * (self.nPos[1])
            self.queryInd = self.queryInd.astype(int)
            # Calculate alpha for cuboid if it doesn't exist
            if self.alphamask[self.queryInd] == 0:
                self.calcCoefficients(self.queryInd)
            # 4-vectors for finding interpolated values
            xvec = np.array([1, cuboidx, cuboidx**2, cuboidx**3])
            yvec = np.array([1, cuboidy, cuboidy**2, cuboidy**3])
            zvec = np.array([1, cuboidz, cuboidz**2, cuboidz**3])
            # 4-vectors for finding interpolated gradients
            xxvec = np.array([0, 1, 2 * cuboidx, 3 * cuboidx**2])
            yyvec = np.array([0, 1, 2 * cuboidy, 3 * cuboidy**2])
            zzvec = np.array([0, 1, 2 * cuboidz, 3 * cuboidz**2])
            # 4-vector summation components
            x = np.tile(xvec, 16)
            y = np.tile(np.repeat(yvec, 4), 4)
            z = np.repeat(zvec, 16)
            # 4-vector summation components
            xx = np.tile(xxvec, 16)
            yy = np.tile(np.repeat(yyvec, 4), 4)
            zz = np.repeat(zzvec, 16)
            # interpolated values
            compx = np.inner(self.alphax[:, self.queryInd], (x * y * z))
            compy = np.inner(self.alphay[:, self.queryInd], (x * y * z))
            compz = np.inner(self.alphaz[:, self.queryInd], (x * y * z))
            norm = np.inner(self.alphan[:, self.queryInd], (x * y * z))
            grad = np.array([np.dot(self.alphan[:, self.queryInd], xx * y * z) / self.hx, np.dot(self.alphan[:, self.queryInd], x * yy * z) / self.hy, np.dot(self.alphan[:, self.queryInd], x * y * zz) / self.hz])
            # Components, magnitude, gradient
            return np.array((compx, compy, compz)), norm, grad

    def _call_array1(self, query):
        # Finds base cuboid indices of the points to be interpolated
        ### Length of sample distribution ###
        N = len(query)

        # Removes particles that are outside of interpolation volume
        query[np.where(query[:, 0] < self.xIntMin)[0]] = np.nan
        query[np.where(query[:, 0] > self.xIntMax)[0]] = np.nan
        query[np.where(query[:, 1] < self.yIntMin)[0]] = np.nan
        query[np.where(query[:, 1] > self.yIntMax)[0]] = np.nan
        query[np.where(query[:, 2] < self.zIntMin)[0]] = np.nan
        query[np.where(query[:, 2] > self.zIntMax)[0]] = np.nan

        # Coords in cuboids
        iu = (query[:, 0] - self.xIntMin) / self.hx
        iv = (query[:, 1] - self.yIntMin) / self.hy
        iw = (query[:, 2] - self.zIntMin) / self.hz

        ### Finds base coordinates of cuboid particles are in ###
        ix = np.floor(iu)
        iy = np.floor(iv)
        iz = np.floor(iw)

        ### Returns indices of base cuboids ###
        self.queryInds = ix + iy * (self.nPos[0]) + iz * (self.nPos[0]) * (self.nPos[1])
        self.queryInds[np.where(np.isnan(self.queryInds))] = self.nc
        self.queryInds = self.queryInds.astype(int)

        # Coordinates of the sample in unit cuboid
        queryCoords = np.stack((iu - ix, iv - iy, iw - iz), axis=1)

        # Returns the interpolate magnitude and / or gradients at the query coordinates
        if len(self.queryInds[np.where(self.alphamask[self.queryInds] == 0)[0]]) > 0:
            list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds] == 0)[0]]))

        # Calculate interpolated values
        x = np.tile(np.transpose(np.array([np.ones(N), queryCoords[:, 0], queryCoords[:, 0]**2, queryCoords[:, 0]**3])), 16)
        y = np.tile(np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:, 1], queryCoords[:, 1]**2, queryCoords[:, 1]**3])), 4, axis=1), 4)
        z = np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:, 2], queryCoords[:, 2]**2, queryCoords[:, 2]**3])), 16, axis=1)

        # Return coefficient matrix values, give NaN for invalid locations
        tx = self.alphax[:, self.queryInds]
        tx = np.transpose(tx)
        ty = self.alphay[:, self.queryInds]
        ty = np.transpose(ty)
        tz = self.alphaz[:, self.queryInds]
        tz = np.transpose(tz)

        # Return components
        compsx = np.reshape((tx * (x * y * z)).sum(axis=1), (N, 1))
        compsy = np.reshape((ty * (x * y * z)).sum(axis=1), (N, 1))
        compsz = np.reshape((tz * (x * y * z)).sum(axis=1), (N, 1))

        return np.hstack((compsx, compsy, compsz))



    def _call_array3(self, query):
        # Finds base cuboid indices of the points to be interpolated
        ### Length of sample distribution ###
        N = len(query)

        # Removes particles that are outside of interpolation volume
        query[np.where(query[:, 0] < self.xIntMin)[0]] = np.nan
        query[np.where(query[:, 0] > self.xIntMax)[0]] = np.nan
        query[np.where(query[:, 1] < self.yIntMin)[0]] = np.nan
        query[np.where(query[:, 1] > self.yIntMax)[0]] = np.nan
        query[np.where(query[:, 2] < self.zIntMin)[0]] = np.nan
        query[np.where(query[:, 2] > self.zIntMax)[0]] = np.nan

        # Coords in cuboids
        iu = (query[:, 0] - self.xIntMin) / self.hx
        iv = (query[:, 1] - self.yIntMin) / self.hy
        iw = (query[:, 2] - self.zIntMin) / self.hz

        ### Finds base coordinates of cuboid particles are in ###
        ix = np.floor(iu)
        iy = np.floor(iv)
        iz = np.floor(iw)

        ### Returns indices of base cuboids ###
        self.queryInds = ix + iy * (self.nPos[0]) + iz * (self.nPos[0]) * (self.nPos[1])
        self.queryInds[np.where(np.isnan(self.queryInds))] = self.nc
        self.queryInds = self.queryInds.astype(int)

        # Coordinates of the sample in unit cuboid
        queryCoords = np.stack((iu - ix, iv - iy, iw - iz), axis=1)

        # Returns the interpolate magnitude and / or gradients at the query coordinates
        if len(self.queryInds[np.where(self.alphamask[self.queryInds] == 0)[0]]) > 0:
            list(map(self.calcCoefficients, self.queryInds[np.where(self.alphamask[self.queryInds] == 0)[0]]))

        # Calculate interpolated values
        x = np.tile(np.transpose(np.array([np.ones(N), queryCoords[:, 0], queryCoords[:, 0]**2, queryCoords[:, 0]**3])), 16)
        y = np.tile(np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:, 1], queryCoords[:, 1]**2, queryCoords[:, 1]**3])), 4, axis=1), 4)
        z = np.repeat(np.transpose(np.array([np.ones(N), queryCoords[:, 2], queryCoords[:, 2]**2, queryCoords[:, 2]**3])), 16, axis=1)

        # Derivatives
        xx = np.tile(np.transpose(np.array([np.zeros(N), np.ones(N), 2 * queryCoords[:, 0], 3 * queryCoords[:, 0]**2])), 16)
        yy = np.tile(np.repeat(np.transpose(np.array([np.zeros(N), np.ones(N), 2 * queryCoords[:, 1], 3 * queryCoords[:, 1]**2])), 4, axis=1), 4)
        zz = np.repeat(np.transpose(np.array([np.zeros(N), np.ones(N), 2 * queryCoords[:, 2], 3 * queryCoords[:, 2]**2])), 16, axis=1)

        # Return coefficient matrix values, give NaN for invalid locations
        tx = self.alphax[:, self.queryInds]
        tx = np.transpose(tx)
        ty = self.alphay[:, self.queryInds]
        ty = np.transpose(ty)
        tz = self.alphaz[:, self.queryInds]
        tz = np.transpose(tz)
        tn = self.alphan[:, self.queryInds]
        tn = np.transpose(tn)

        # Return components and magnitude
        compsx = np.reshape((tx * (x * y * z)).sum(axis=1), (N, 1))
        compsy = np.reshape((ty * (x * y * z)).sum(axis=1), (N, 1))
        compsz = np.reshape((tz * (x * y * z)).sum(axis=1), (N, 1))
        norms = np.reshape((tn * (x * y * z)).sum(axis=1), (N, 1))

        # Return gradient
        grads = np.transpose(np.array([((tn * (xx * y * z)) / self.hx).sum(axis=1), ((tn * (x * yy * z)) / self.hy).sum(axis=1), ((tn * (x * y * zz)) / self.hz).sum(axis=1)]))

        return np.hstack((compsx, compsy, compsz)), norms, grads


    def calcCoefficients1(self, alphaindex):
        if self.alphamask[alphaindex] == 0:
            # Find interpolation coefficients for a cuboid
            realindex = self.basePointInds[alphaindex]
            # Find other vertices of current cuboid, and all neighbours in 3x3x3 neighbouring array
            inds = self.neighbourInd(realindex)
            # Alpha coefficients
            self.alphax[:, alphaindex] = np.dot(self.A, self.Bx[inds])
            self.alphay[:, alphaindex] = np.dot(self.A, self.By[inds])
            self.alphaz[:, alphaindex] = np.dot(self.A, self.Bz[inds])
            self.alphamask[alphaindex] = 1

   
    def calcCoefficients3(self, alphaindex):
        if self.alphamask[alphaindex] == 0:
            # Find interpolation coefficients for a cuboid
            realindex = self.basePointInds[alphaindex]
            # Find other vertices of current cuboid, and all neighbours in 3x3x3 neighbouring array
            inds = self.neighbourInd(realindex)
            # Alpha coefficients
            self.alphax[:, alphaindex] = np.dot(self.A, self.Bx[inds])
            self.alphay[:, alphaindex] = np.dot(self.A, self.By[inds])
            self.alphaz[:, alphaindex] = np.dot(self.A, self.Bz[inds])
            self.alphan[:, alphaindex] = np.dot(self.A, self.Bn[inds])
            self.alphamask[alphaindex] = 1
"""
