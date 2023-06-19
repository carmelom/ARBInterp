#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Created: 10/2022
# Author: Carmelo Mordini <cmordini@phys.ethz.ch>

from pathlib import Path
import numpy as np

file_path = Path(__file__).parent
matrix_path = file_path / "scaled_by8_inverse_matrix.csv"

class TricubicInterpolatorBase:

    def __init__(self, field, *args, **kwargs):
        self.eps = 0.1 * np.finfo(float).eps  # Machine precision of floating point number
        # Load field passed to class - can be x,y,z or norm of vector field, or scalar
        self.inputfield = field
        # Analyse field, get shapes etc
        self.getFieldParams()
        # Make coefficient matrix
        self.A, self.D = self.makeAMatrix()

        # Mask to identify where coefficients exist
        self.alphamask = np.zeros((self.nc + 1, 1))
        self.alphamask[-1] = 1

    def __call__(self, xi, d=0):
        if d == 0:
            return self.field(xi)
        elif d == 1:
            return self.gradient(xi)
        elif d == 2:
            return self.hessian(xi)
        else:
            raise NotImplementedError

    def field(self, xi):
        return NotImplementedError

    def gradient(self, xi):
        return NotImplementedError

    def hessian(self, xi):
        return NotImplementedError

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

        # A = np.matmul(np.linalg.inv(B), D)
        A = np.linalg.inv(B)
        return A, D

    def neighbourInd(self, ind0):
        # For base index ind0 this finds all 64 vertices of the 3x3x3 range of cuboids around it
        # It also returns the 7 neighbouring points
        # newind0 = ind0 - 1 - (self.nPos[0] + 3) * (self.nPos[1] + 4)
        # bInds = np.zeros(64)
        # bInds[0] = newind0
        # bInds[1] = bInds[0] + 1
        # bInds[2] = bInds[1] + 1
        # bInds[3] = bInds[2] + 1
        # bInds[4:8] = bInds[:4] + self.nPos[0] + 3
        # bInds[8:12] = bInds[4:8] + self.nPos[0] + 3
        # bInds[12:16] = bInds[8:12] + self.nPos[0] + 3
        # bInds[16:32] = bInds[:16] + (self.nPos[0] + 3) * (self.nPos[1] + 3)
        # bInds[32:48] = bInds[16:32] + (self.nPos[0] + 3) * (self.nPos[1] + 3)
        # bInds[48:] = bInds[32:48] + (self.nPos[0] + 3) * (self.nPos[1] + 3)
        # bInds = bInds.astype(int)
        bInds = np.zeros(8)
        bInds[0] = ind0
        bInds[1] = ind0 + 1
        bInds[2:4] = bInds[0:2] + self.nPos[0] + 3
        bInds[4:8] = bInds[:4] + (self.nPos[1] + 3) * (self.nPos[0] + 3)
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
        query[np.where(query[:, 0] >= self.xIntMax)[0]] = np.nan
        query[np.where(query[:, 1] < self.yIntMin)[0]] = np.nan
        query[np.where(query[:, 1] >= self.yIntMax)[0]] = np.nan
        query[np.where(query[:, 2] < self.zIntMin)[0]] = np.nan
        query[np.where(query[:, 2] >= self.zIntMax)[0]] = np.nan
        return query


class TriquinticInterpolatorBase:

    def __init__(self, field, *args, **kwargs):
        self.eps = 0.1 * np.finfo(float).eps  # Machine precision of floating point number
        # Load field passed to class - can be x,y,z or norm of vector field, or scalar
        self.inputfield = field
        # Analyse field, get shapes etc
        self.getFieldParams()
        # Make coefficient matrix
        self.A = self.makeAMatrix()

        # Mask to identify where coefficients exist
        self.alphamask = np.zeros((self.nc + 1, 1))
        self.alphamask[-1] = 1

    def __call__(self, xi, d=0):
        if d == 0:
            return self.field(xi)
        elif d == 1:
            return self.gradient(xi)
        elif d == 2:
            return self.hessian(xi)
        else:
            raise NotImplementedError

    def field(self, xi):
        return NotImplementedError

    def gradient(self, xi):
        return NotImplementedError

    def hessian(self, xi):
        return NotImplementedError

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

        # Compared to tricubic (nPosx-3) we now have (nPosx-5) because of higher derivatives have
        # to be approximated
        self.nPos = np.array([nPosx - 5, nPosy - 5, nPosz - 5])  # number of interpolatable cuboids per axis

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

        # Compared to tricubic, we use the first/last 2 points of the partition for estimating
        # derivatives
        self.xIntMin = self.inputfield[2, 0]  # Minimal value of x that can be interpolated
        self.yIntMin = self.inputfield[2 * nPosx, 1]
        self.zIntMin = self.inputfield[2 * nPosx * nPosy, 2]
        self.xIntMax = self.inputfield[-3, 0]  # Maximal value of x that can be interpolated
        self.yIntMax = self.inputfield[-2 * nPosx - 1, 1]
        self.zIntMax = self.inputfield[-2 * nPosx * nPosy - 1, 2]

        # Find base indices of all interpolatable cuboids
        minI = 2 * nPosx * nPosy + 2 * nPosx + 2
        self.basePointInds = minI + np.arange(0, nPosx - 5, 1)
        temp = np.array([self.basePointInds + i * nPosx for i in range(nPosy - 5)])
        self.basePointInds = np.reshape(temp, (1, len(temp) * len(temp[0])))[0]
        temp = np.array([self.basePointInds + i * nPosx * nPosy for i in range(nPosz - 5)])
        self.basePointInds = np.reshape(temp, (1, len(temp) * len(temp[0])))[0]
        self.basePointInds = np.sort(self.basePointInds)

        # Number of interpolatable cuboids
        self.nc = len(self.basePointInds)

    def makeAMatrix(self):
        # Loads the triquintic interpolation matrix and creates finite difference matrix and combines them
        # Interpolation matrix (inverse)
        Bm1_scaled_by_8 = np.genfromtxt(matrix_path, delimiter=',')
        Bm1 = Bm1_scaled_by_8 / 8

        # This makes a finite-difference matrix to return the components of the "b"-vector
        # needed in the alpha calculation, in "cuboid" coordinates
        # In the triquintic approximation we have a unit cube whose main axes are extended by two points
        # in each direction ending up in a 6x6x6 grid (in tricubic the extension was by one point in each
        # direction ending up in a 4x4x4 grid)
        C = np.array((86, 87, 92, 93, 122, 123, 128, 129))
        D = np.zeros((216, 216))

        # Scalar values at cuboid vertices
        for i in range(8):
            D[i, C[i]] = 1

        # x-derivative at cuboid vertices
        for i, j in enumerate(range(8, 16, 1)):
            D[j, C[i] - 1 + 0 + 0] = -1/2
            D[j, C[i] + 1 + 0 + 0] = 1/2

        # y-derivative at cuboid vertices
        for i, j in enumerate(range(16, 24, 1)):
            D[j, C[i] + 0 - 6 + 0] = -1/2
            D[j, C[i] + 0 + 6 + 0] = 1/2

        # z-derivative at cuboid vertices
        for i, j in enumerate(range(24, 32, 1)):
            D[j, C[i] + 0 + 0 - 36] = -1/2
            D[j, C[i] + 0 + 0 + 36] = 1/2

        # xx-derivative at cuboid vertices
        for i, j in enumerate(range(32, 40, 1)):
            D[j, C[i] + 2 + 0 + 0] = 1/4
            D[j, C[i] + 0 + 0 + 0] = -1/2
            D[j, C[i] - 2 + 0 + 0] = 1/4

        # yx-derivative at cuboid vertices
        for i, j in enumerate(range(40, 48, 1)):
            D[j, C[i] + 1 + 6 + 0] = 1/4
            D[j, C[i] + 1 - 6 + 0] = -1/4
            D[j, C[i] - 1 + 6 + 0] = -1/4
            D[j, C[i] - 1 - 6 + 0] = 1/4

        # zx-derivative at cuboid vertices
        for i, j in enumerate(range(48, 56, 1)):
            D[j, C[i] + 1 + 0 + 36] = 1/4
            D[j, C[i] + 1 + 0 - 36] = -1/4
            D[j, C[i] - 1 + 0 + 36] = -1/4
            D[j, C[i] - 1 + 0 - 36] = 1/4

        # yy-derivative at cuboid vertices
        for i, j in enumerate(range(56, 64, 1)):
            D[j, C[i] + 0 + 12 + 0] = 1/4
            D[j, C[i] + 0 + 0 + 0] = -1/2
            D[j, C[i] + 0 - 12 + 0] = 1/4

        # zy-derivative at cuboid vertices
        for i, j in enumerate(range(64, 72, 1)):
            D[j, C[i] + 0 + 6 + 36] = 1/4
            D[j, C[i] + 0 + 6 - 36] = -1/4
            D[j, C[i] + 0 - 6 + 36] = -1/4
            D[j, C[i] + 0 - 6 - 36] = 1/4

        # yy-derivative at cuboid vertices
        for i, j in enumerate(range(72, 80, 1)):
            D[j, C[i] + 0 + 0 + 64] = 1/4
            D[j, C[i] + 0 + 0 + 0] = -1/2
            D[j, C[i] + 0 + 0 - 64] = 1/4

        # yxx-derivative at cuboid vertices
        for i, j in enumerate(range(80, 88, 1)):
            D[j, C[i] + 2 + 6 + 0] = 1/8
            D[j, C[i] + 2 - 6 + 0] = -1/8
            D[j, C[i] + 0 + 6 + 0] = -1/4
            D[j, C[i] + 0 - 6 + 0] = 1/4
            D[j, C[i] - 2 + 6 + 0] = 1/8
            D[j, C[i] - 2 - 6 + 0] = -1/8
        
        # zxx-derivative at cuboid vertices
        for i, j in enumerate(range(88, 96, 1)):
            D[j, C[i] + 2 + 0 + 36] = 1/8
            D[j, C[i] + 2 + 0 - 36] = -1/8
            D[j, C[i] + 0 + 0 + 36] = -1/4
            D[j, C[i] + 0 + 0 - 36] = 1/4
            D[j, C[i] - 2 + 0 + 36] = 1/8
            D[j, C[i] - 2 + 0 - 36] = -1/8
        
        # yyx-derivative at cuboid vertices
        for i, j in enumerate(range(96, 104, 1)):
            D[j, C[i] + 1 + 12 + 0] = 1/8
            D[j, C[i] + 1 + 0 + 0] = -1/4
            D[j, C[i] + 1 - 12 + 0] = 1/8
            D[j, C[i] - 1 + 12 + 0] = -1/8
            D[j, C[i] - 1 + 0 + 0] = 1/4
            D[j, C[i] - 1 - 12 + 0] = -1/8
        
        # zyx-derivative at cuboid vertices
        for i, j in enumerate(range(104, 112, 1)):
            D[j, C[i] + 1 + 6 + 36] = 1/8
            D[j, C[i] + 1 + 6 - 36] = -1/8
            D[j, C[i] + 1 - 6 + 36] = -1/8
            D[j, C[i] + 1 - 6 - 36] = 1/8
            D[j, C[i] - 1 + 6 + 36] = -1/8
            D[j, C[i] - 1 + 6 - 36] = 1/8
            D[j, C[i] - 1 - 6 + 36] = 1/8
            D[j, C[i] - 1 - 6 - 36] = -1/8
        
        # zyy-derivative at cuboid vertices
        for i, j in enumerate(range(112, 120, 1)):
            D[j, C[i] + 0 + 12 + 36] = 1/8
            D[j, C[i] + 0 + 12 - 36] = -1/8
            D[j, C[i] + 0 + 0 + 36] = -1/4
            D[j, C[i] + 0 + 0 - 36] = 1/4
            D[j, C[i] + 0 - 12 + 36] = 1/8
            D[j, C[i] + 0 - 12 - 36] = -1/8

        # zzx-derivative at cuboid vertices
        for i, j in enumerate(range(120, 128, 1)):
            D[j, C[i] + 1 + 0 + 64] = 1/8
            D[j, C[i] + 1 + 0 + 0] = -1/4
            D[j, C[i] + 1 + 0 - 64] = 1/8
            D[j, C[i] - 1 + 0 + 64] = -1/8
            D[j, C[i] - 1 + 0 + 0] = 1/4
            D[j, C[i] - 1 + 0 - 64] = -1/8

        # zzy-derivative at cuboid vertices
        for i, j in enumerate(range(128, 136, 1)):
            D[j, C[i] + 0 + 6 + 64] = 1/8
            D[j, C[i] + 0 + 6 + 0] = -1/4
            D[j, C[i] + 0 + 6 - 64] = 1/8
            D[j, C[i] + 0 - 12 + 64] = -1/8
            D[j, C[i] + 0 - 12 + 0] = 1/4
            D[j, C[i] + 0 - 12 - 64] = -1/8

        # yyxx-derivative at cuboid vertices
        for i, j in enumerate(range(136, 144, 1)):
            D[j, C[i] + 2 + 12 + 0] = 1/16
            D[j, C[i] + 2 + 0 + 0] = -1/8
            D[j, C[i] + 2 - 12 - 0] = 1/16
            D[j, C[i] + 0 + 12 + 0] = -1/8
            D[j, C[i] + 0 + 0 + 0] = 1/4
            D[j, C[i] + 0 - 12 - 0] = -1/8
            D[j, C[i] - 2 + 12 + 0] = 1/16
            D[j, C[i] - 2 + 0 + 0] = -1/8
            D[j, C[i] - 2 - 12 - 0] = 1/16
        
        # zzxx-derivative at cuboid vertices
        for i, j in enumerate(range(144, 152, 1)):
            D[j, C[i] + 2 + 0 + 72] = 1/16
            D[j, C[i] + 2 + 0 + 0] = -1/8
            D[j, C[i] + 2 - 0 - 72] = 1/16
            D[j, C[i] + 0 + 0 + 72] = -1/8
            D[j, C[i] + 0 + 0 + 0] = 1/4
            D[j, C[i] + 0 - 0 - 72] = -1/8
            D[j, C[i] - 2 + 0 + 72] = 1/16
            D[j, C[i] - 2 + 0 + 0] = -1/8
            D[j, C[i] - 2 - 0 - 72] = 1/16

        # zzyy-derivative at cuboid vertices
        for i, j in enumerate(range(152, 160, 1)):
            D[j, C[i] + 0 + 12 + 72] = 1/16
            D[j, C[i] + 0 + 12 + 0] = -1/8
            D[j, C[i] + 0 + 12 - 72] = 1/16
            D[j, C[i] + 0 + 0 + 72] = -1/8
            D[j, C[i] + 0 + 0 + 0] = 1/4
            D[j, C[i] + 0 - 0 - 72] = -1/8
            D[j, C[i] + 0 - 12 + 72] = 1/16
            D[j, C[i] + 0 - 12 + 0] = -1/8
            D[j, C[i] + 0 - 12 - 72] = 1/16
        
        # zyxx-derivative at cuboid vertices
        for i, j in enumerate(range(160, 168, 1)):
            D[j, C[i] + 2 + 6 + 36] = 1/16
            D[j, C[i] + 2 - 6 + 36] = -1/16
            D[j, C[i] + 2 + 6 - 36] = -1/16
            D[j, C[i] + 2 - 6 - 36] = 1/16
            D[j, C[i] + 0 + 6 + 36] = -1/8
            D[j, C[i] + 0 - 6 + 36] = 1/8
            D[j, C[i] + 0 + 6 - 36] = 1/8
            D[j, C[i] + 0 - 6 - 36] = -1/8
            D[j, C[i] - 2 + 6 + 36] = 1/16
            D[j, C[i] - 2 - 6 + 36] = -1/16
            D[j, C[i] - 2 + 6 - 36] = -1/16
            D[j, C[i] - 2 - 6 - 36] = 1/16
        
        # zyyx-derivative at cuboid vertices
        for i, j in enumerate(range(168, 176, 1)):
            D[j, C[i] + 1 + 12 + 36] = 1/16
            D[j, C[i] + 1 + 12 - 36] = -1/16
            D[j, C[i] + 1 + 0 + 36] = -1/8
            D[j, C[i] + 1 + 0 - 36] = 1/8
            D[j, C[i] + 1 - 12 + 36] = 1/16
            D[j, C[i] + 1 - 12 - 36] = -1/16
            D[j, C[i] - 1 + 12 + 36] = -1/16
            D[j, C[i] - 1 + 12 - 36] = 1/16
            D[j, C[i] - 1 + 0 + 36] = 1/8
            D[j, C[i] - 1 + 0 - 36] = -1/8
            D[j, C[i] - 1 - 12 + 36] = -1/16
            D[j, C[i] - 1 - 12 - 36] = 1/16

        # zzyx-derivative at cuboid vertices
        for i, j in enumerate(range(176, 184, 1)):
            D[j, C[i] + 1 + 6 + 72] = 1/16
            D[j, C[i] + 1 + 6 + 0] = -1/8
            D[j, C[i] + 1 + 6 - 72] = 1/16
            D[j, C[i] + 1 - 6 + 72] = -1/16
            D[j, C[i] + 1 - 6 + 0] = 1/8
            D[j, C[i] + 1 - 6 - 72] = -1/16
            D[j, C[i] - 1 + 6 + 72] = -1/16
            D[j, C[i] - 1 + 6 + 0] = 1/8
            D[j, C[i] - 1 + 6 - 72] = -1/16
            D[j, C[i] - 1 - 6 + 72] = 1/16
            D[j, C[i] - 1 - 6 + 0] = -1/8
            D[j, C[i] - 1 - 6 - 72] = 1/16
        
        # zyyxx-derivative at cuboid vertices
        for i, j in enumerate(range(184, 192, 1)):
            D[j, C[i] + 2 + 12 + 36] = 1/32
            D[j, C[i] + 2 + 12 - 36] = -1/32
            D[j, C[i] + 2 + 0 + 36] = -1/16
            D[j, C[i] + 2 + 0 - 36] = 1/16
            D[j, C[i] + 2 - 12 + 36] = 1/32
            D[j, C[i] + 2 - 12 - 36] = -1/32
            D[j, C[i] + 0 + 12 + 36] = -1/16
            D[j, C[i] + 0 + 12 - 36] = 1/16
            D[j, C[i] + 0 + 0 + 36] = 1/8
            D[j, C[i] + 0 + 0 - 36] = -1/8
            D[j, C[i] + 0 - 12 + 36] = -1/16
            D[j, C[i] + 0 - 12 - 36] = 1/16
            D[j, C[i] - 2 + 12 + 36] = 1/32
            D[j, C[i] - 2 + 12 - 36] = -1/32
            D[j, C[i] - 2 + 0 + 36] = -1/16
            D[j, C[i] - 2 + 0 - 36] = 1/16
            D[j, C[i] - 2 - 12 + 36] = 1/32
            D[j, C[i] - 2 - 12 - 36] = -1/32

        # zzyxx-derivative at cuboid vertices
        for i, j in enumerate(range(192, 200, 1)):
            D[j, C[i] + 2 + 6 + 72] = 1/32
            D[j, C[i] + 2 + 6 + 0] = -1/16
            D[j, C[i] + 2 + 6 - 72] = 1/32
            D[j, C[i] + 2 - 6 + 72] = -1/32
            D[j, C[i] + 2 - 6 + 0] = 1/16
            D[j, C[i] + 2 - 6 - 72] = -1/32
            D[j, C[i] + 0 + 6 + 72] = -1/16
            D[j, C[i] + 0 + 6 + 0] = 1/8
            D[j, C[i] + 0 + 6 - 72] = -1/16
            D[j, C[i] + 0 - 6 + 72] = 1/16
            D[j, C[i] + 0 - 6 + 0] = -1/8
            D[j, C[i] + 0 - 6 - 72] = 1/16
            D[j, C[i] - 2 + 6 + 72] = 1/32
            D[j, C[i] - 2 + 6 + 0] = -1/16
            D[j, C[i] - 2 + 6 - 72] = 1/32
            D[j, C[i] - 2 - 6 + 72] = -1/32
            D[j, C[i] - 2 - 6 + 0] = 1/16
            D[j, C[i] - 2 - 6 - 72] = -1/32

        # zzyyx-derivative at cuboid vertices
        for i, j in enumerate(range(200, 208, 1)):
            D[j, C[i] + 1 + 12 + 72] = 1/32
            D[j, C[i] + 1 + 12 + 0] = -1/16
            D[j, C[i] + 1 + 12 - 72] = 1/32
            D[j, C[i] + 1 + 0 + 72] = -1/16
            D[j, C[i] + 1 + 0 + 0] = 1/8
            D[j, C[i] + 1 + 0 - 72] = -1/16
            D[j, C[i] + 1 - 12 + 72] = 1/32
            D[j, C[i] + 1 - 12 + 0] = -1/16
            D[j, C[i] + 1 - 12 - 72] = 1/32
            D[j, C[i] - 1 + 12 + 72] = -1/32
            D[j, C[i] - 1 + 12 + 0] = 1/16
            D[j, C[i] - 1 + 12 - 72] = -1/32
            D[j, C[i] - 1 + 0 + 72] = 1/16
            D[j, C[i] - 1 + 0 + 0] = -1/8
            D[j, C[i] - 1 + 0 - 72] = 1/16
            D[j, C[i] - 1 - 12 + 72] = -1/32
            D[j, C[i] - 1 - 12 + 0] = 1/16
            D[j, C[i] - 1 - 12 - 72] = -1/32

        # zzyyxx-derivative at cuboid vertices
        for i, j in enumerate(range(208, 216, 1)):
            D[j, C[i] + 2 + 12 + 72] = 1/64
            D[j, C[i] + 2 + 12 + 0] = -1/32
            D[j, C[i] + 2 + 12 - 72] = 1/64
            D[j, C[i] + 2 + 0 + 72] = -1/32
            D[j, C[i] + 2 + 0 + 0] = 1/16
            D[j, C[i] + 2 + 0 - 72] = -1/32
            D[j, C[i] + 2 - 12 + 72] = 1/64
            D[j, C[i] + 2 - 12 + 0] = -1/32
            D[j, C[i] + 2 - 12 - 72] = 1/64
            D[j, C[i] + 0 + 12 + 72] = -1/32
            D[j, C[i] + 0 + 12 + 0] = 1/16
            D[j, C[i] + 0 + 12 - 72] = -1/32
            D[j, C[i] + 0 + 0 + 72] = 1/16
            D[j, C[i] + 0 + 0 + 0] = -1/8
            D[j, C[i] + 0 + 0 - 72] = 1/16
            D[j, C[i] + 0 - 12 + 72] = -1/32
            D[j, C[i] + 0 - 12 + 0] = 1/16
            D[j, C[i] + 0 - 12 - 72] = -1/32
            D[j, C[i] - 2 + 12 + 72] = 1/64
            D[j, C[i] - 2 + 12 + 0] = -1/32
            D[j, C[i] - 2 + 12 - 72] = 1/64
            D[j, C[i] - 2 + 0 + 72] = -1/32
            D[j, C[i] - 2 + 0 + 0] = 1/16
            D[j, C[i] - 2 + 0 - 72] = -1/32
            D[j, C[i] - 2 - 12 + 72] = 1/64
            D[j, C[i] - 2 - 12 + 0] = -1/32
            D[j, C[i] - 2 - 12 - 72] = 1/64
        

        A = np.matmul(Bm1, D)
        return A

    def neighbourInd(self, ind0):
        # For base index ind0 this finds all 216 vertices of the 5x5x5 range of cuboids around it
        # It also returns the 7 neighbouring points
        newind0 = ind0 - 2 - 2 * (self.nPos[0] + 5) * (self.nPos[1] + 5) - 2 * (self.nPos[1] + 5)
        bInds = np.zeros(216)
        bInds[0] = newind0
        bInds[1] = bInds[0] + 1
        bInds[2] = bInds[1] + 1
        bInds[3] = bInds[2] + 1
        bInds[4] = bInds[3] + 1
        bInds[5] = bInds[4] + 1
        bInds[6:12] = bInds[:6] + self.nPos[0] + 5
        bInds[12:18] = bInds[6:12] + self.nPos[0] + 5
        bInds[18:24] = bInds[12:18] + self.nPos[0] + 5
        bInds[24:30] = bInds[18:24] + self.nPos[0] + 5
        bInds[30:36] = bInds[24:30] + self.nPos[0] + 5
        bInds[36:72] = bInds[:36] + (self.nPos[0] + 5) * (self.nPos[1] + 5)
        bInds[72:108] = bInds[36:72] + (self.nPos[0] + 5) * (self.nPos[1] + 5)
        bInds[108:144] = bInds[72:108] + (self.nPos[0] + 5) * (self.nPos[1] + 5)
        bInds[144:180] = bInds[108:144] + (self.nPos[0] + 5) * (self.nPos[1] + 5)
        bInds[180:216] = bInds[144:180] + (self.nPos[0] + 5) * (self.nPos[1] + 5)
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
