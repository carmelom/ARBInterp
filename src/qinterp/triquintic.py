import numpy as np
from .base import TriquinticInterpolatorBase


class TriquinticScalarInterpolator(TriquinticInterpolatorBase):

    def __init__(self, field, *args, **kwargs):
        super().__init__(field, *args, **kwargs)
        self.alphan = np.zeros((216, self.nc + 1))
        self.alphan[:, -1] = np.nan
        self.Bn = self.inputfield[:, 3]
        # precalculate all coefficients
        # self.allCoeffs()

    def calcCoefficients(self, alphaindex):
        # Find interpolation coefficients for a cuboid
        realindex = self.basePointInds[alphaindex]
        # Find other vertices of current cuboid, and all neighbours in 3x3x3 neighbouring array
        inds = self.neighbourInd(realindex)
        # Alpha coefficients
        self.alphan[:, alphaindex] = np.dot(self.A, self.Bn[inds])
        self.alphamask[alphaindex] = 1

    def field(self, xi):
        if xi.ndim == 1:
            return self._field1(xi)
        return self._field(xi)

    def gradient(self, xi):
        if xi.ndim == 1:
            return self._gradient1(xi)
        return self._gradient(xi)

    def hessian(self, xi):
        if xi.ndim == 1:
            return self._hessian1(xi)
        return self._hessian(xi)

    def _field1(self, point):
        """ Calculate field and derivatives at one single point
        """
        if self.check_out_of_bounds(point):
            return np.nan

        queryIndex, cx, cy, cz = self._cuboid_coordinates1(point)
        tn = self.alphan[:, queryIndex]

        x, y, z = self._pack_coords1(cx, cy, cz, d=0)

        norm = np.inner(tn, (x * y * z))
        return norm

    def _gradient1(self, point):
        """ Calculate gradient at one single point
        """
        if self.check_out_of_bounds(point):
            return np.empty((3,)) * np.nan

        queryIndex, cx, cy, cz = self._cuboid_coordinates1(point)
        tn = self.alphan[:, queryIndex]

        x, y, z = self._pack_coords1(cx, cy, cz, d=0)
        xx, yy, zz = self._pack_coords1(cx, cy, cz, d=1)

        grad = np.array([np.dot(tn, xx * y * z) / self.hx, np.dot(tn, x * yy * z) / self.hy, np.dot(tn, x * y * zz) / self.hz])
        return grad

    def _hessian1(self, point):
        """ Calculate curvature at one single point
        """
        if self.check_out_of_bounds(point):
            return np.empty((3, 3)) * np.nan

        queryIndex, cx, cy, cz = self._cuboid_coordinates1(point)
        tn = self.alphan[:, queryIndex]

        x, y, z = self._pack_coords1(cx, cy, cz, d=0)
        xx, yy, zz = self._pack_coords1(cx, cy, cz, d=1)
        xxx, yyy, zzz = self._pack_coords1(cx, cy, cz, d=2)

        hxx = np.dot(tn, xxx * y * z) / self.hx / self.hx
        hxy = np.dot(tn, xx * yy * z) / self.hy / self.hx
        hxz = np.dot(tn, xx * y * zz) / self.hz / self.hx
        hyy = np.dot(tn, x * yyy * z) / self.hy / self.hy
        hyz = np.dot(tn, x * yy * zz) / self.hz / self.hy
        hzz = np.dot(tn, x * y * zzz) / self.hz / self.hz
        hess = np.array([
            [hxx, hxy, hxz],
            [hxy, hyy, hyz],
            [hxz, hyz, hzz],
        ])
        return hess

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
        cx = iu - ix
        cy = iv - iy
        cz = iw - iz
        # Returns index of base cuboid
        queryIndex = ix + iy * (self.nPos[0]) + iz * (self.nPos[0]) * (self.nPos[1])
        queryIndex = queryIndex.astype(int)

        # Calculate alpha for cuboid if it doesn't exist
        if self.alphamask[queryIndex] == 0:
            self.calcCoefficients(queryIndex)

        return queryIndex, cx, cy, cz

    def _pack_coords1(self, cx, cy, cz, d=0):
        if d == 0:
            x = np.tile(np.array([1, cx, cx**2, cx**3, cx**4, cx**5]), 36)
            y = np.tile(np.repeat(np.array([1, cy, cy**2, cy**3, cy**4, cy**5]), 6), 6)
            z = np.repeat(np.array([1, cz, cz**2, cz**3, cz**4, cz**5]), 36)
        elif d == 1:
            # 4-vectors for finding interpolated gradient
            x = np.tile(np.array([0, 1, 2 * cx, 3 * cx**2, 4 * cx**3, 5 * cx**4]), 36)
            y = np.tile(np.repeat(np.array([0, 1, 2 * cy, 3 * cy**2, 4 * cy**3, 5 * cy**4]), 6), 6)
            z = np.repeat(np.array([0, 1, 2 * cz, 3 * cz**2, 4 * cz**3, 5 * cz**4]), 36)
        elif d == 2:
            # 4-vectors for finding interpolated hessian
            x = np.tile(np.array([0, 0, 2, 6 * cx, 12 * cx**2, 20 * cx**3]), 36)
            y = np.tile(np.repeat(np.array([0, 0, 2, 6 * cy, 12 * cy**2, 20 * cy**3]), 6), 6)
            z = np.repeat(np.array([0, 0, 2, 6 * cz, 12 * cz**2, 20 * cz**3]), 36)
        else:
            raise NotImplementedError
        return x, y, z

    def _field(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, cx, cy, cz = self._cuboid_coordinates(points)

        x, y, z = self._pack_coords(cx, cy, cz, d=0)
        # Return coefficient matrix values, give NaN for invalid locations
        tn = self.alphan[:, queryInds]
        tn = np.transpose(tn)

        field = np.reshape((tn * (x * y * z)).sum(axis=1), (-1, 1))
        return field

    def _gradient(self, points):
        points = self.nan_out_of_bounds(points)
        queryInds, cx, cy, cz = self._cuboid_coordinates(points)

        x, y, z = self._pack_coords(cx, cy, cz, d=0)
        xx, yy, zz = self._pack_coords(cx, cy, cz, d=1)

        # Return coefficient matrix values, give NaN for invalid locations
        tn = self.alphan[:, queryInds]
        tn = np.transpose(tn)
        grad = np.transpose(np.array([((tn * (xx * y * z)) / self.hx).sum(axis=1), ((tn * (x * yy * z)) / self.hy).sum(axis=1), ((tn * (x * y * zz)) / self.hz).sum(axis=1)]))
        return grad

    def _hessian(self, points):
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

        # Finds base coordinates of cuboid particles are in ###
        ix = np.floor(iu)
        iy = np.floor(iv)
        iz = np.floor(iw)

        # Returns indices of base cuboids ###
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
            x = np.tile(np.transpose(np.array([one, cx, cx**2, cx**3, cx**4, cx**5])), 36)
            y = np.tile(np.repeat(np.transpose(np.array([one, cy, cy**2, cy**3, cy**4, cy**5])), 6, axis=1), 6)
            z = np.repeat(np.transpose(np.array([one, cz, cz**2, cz**3, cz**4, cz**5])), 36, axis=1)
        elif d == 1:
            # Derivatives
            x = np.tile(np.transpose(np.array([zero, one, 2 * cx, 3 * cx**2, 4 * cx**3, 5 * cx**4])), 36)
            y = np.tile(np.repeat(np.transpose(np.array([zero, one, 2 * cy, 3 * cy**2, 4 * cy**3, 5 * cy**4])), 6, axis=1), 6)
            z = np.repeat(np.transpose(np.array([zero, one, 2 * cz, 3 * cz**2, 4 * cz**3, 5 * cz**4])), 36, axis=1)
        elif d == 2:
            x = np.tile(np.transpose(np.array([zero, zero, 2 * one, 6 * cx, 12 * cx**2, 20 * cx**3])), 36)
            y = np.tile(np.repeat(np.transpose(np.array([zero, zero, 2 * one, 6 * cy, 12 * cy**2, 20 * cy**3])), 6, axis=1), 6)
            z = np.repeat(np.transpose(np.array([zero, zero, 2 * one, 6 * cz, 12 * cz**2, 20 * cz**3])), 36, axis=1)
        else:
            raise NotImplementedError
        return x, y, z


class TriquinticVectorInterpolator(TriquinticInterpolatorBase):

    def __init__(self, field, *args, **kwargs):
        raise NotImplementedError
