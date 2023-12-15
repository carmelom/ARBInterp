import numpy as np
import findiff as fd
from .base import TriquinticInterpolatorBase

_ACC = 6

class TriquinticScalarInterpolator(TriquinticInterpolatorBase):

    def __init__(self, field, accuracy=_ACC, *args, **kwargs):
        super().__init__(field, *args, **kwargs)
        self.alphan = np.zeros((216, self.nc + 1))
        self.alphan[:, -1] = np.nan
        # self.Bn = self.inputfield[:, 3]
        # Transform the field to be a 3D numpy array
        nPosx = self.nPos[0] + 5
        nPosy = self.nPos[1] + 5
        nPosz = self.nPos[2] + 5
        self.f3D = np.reshape(self.inputfield[:,3], (nPosx, nPosy, nPosz), order='F')
        ACC = accuracy
        d_dx = fd.FinDiff(0, 1, 1, acc=ACC)
        d_dy = fd.FinDiff(1, 1, 1, acc=ACC)
        d_dz = fd.FinDiff(2, 1, 1, acc=ACC)
        d2_dx2 = fd.FinDiff(0, 1, 2, acc=ACC)
        d2_dxdy = fd.FinDiff((0, 1, 1), (1, 1, 1), acc=ACC)
        d2_dxdz = fd.FinDiff((0, 1, 1), (2, 1, 1), acc=ACC)
        d2_dy2 = fd.FinDiff(1, 1, 2, acc=ACC)
        d2_dydz = fd.FinDiff((1, 1, 1), (2, 1, 1), acc=ACC)
        d2_dz2 = fd.FinDiff(2, 1, 2, acc=ACC)
        d3_dx2dy = fd.FinDiff((0, 1, 2), (1, 1, 1), acc=ACC)
        d3_dx2dz = fd.FinDiff((0, 1, 2), (2, 1, 1), acc=ACC)
        d3_dxdy2 = fd.FinDiff((0, 1, 1), (1, 1, 2), acc=ACC)
        d3_dxdydz = fd.FinDiff((0, 1, 1), (1, 1, 1), (2, 1, 1), acc=ACC)
        d3_dy2dz = fd.FinDiff((1, 1, 2), (2, 1, 1), acc=ACC)
        d3_dxdz2 = fd.FinDiff((0, 1, 1), (2, 1, 2), acc=ACC)
        d3_dydz2 = fd.FinDiff((1, 1, 1), (2, 1, 2), acc=ACC)
        d4_dx2dy2 = fd.FinDiff((0, 1, 2), (1, 1, 2), acc=ACC)
        d4_dx2dz2 = fd.FinDiff((0, 1, 2), (2, 1, 2), acc=ACC)
        d4_dy2dz2 = fd.FinDiff((1, 1, 2), (2, 1, 2), acc=ACC)
        d4_dx2dydz = fd.FinDiff((0, 1, 2), (1, 1, 1), (2, 1, 1), acc=ACC)
        d4_dxdy2dz = fd.FinDiff((0, 1, 1), (1, 1, 2), (2, 1, 1), acc=ACC)
        d4_dxdydz2 = fd.FinDiff((0, 1, 1), (1, 1, 1), (2, 1, 2), acc=ACC)
        d5_dx2dy2dz = fd.FinDiff((0, 1, 2), (1, 1, 2), (2, 1, 1), acc=ACC)
        d5_dx2dydz2 = fd.FinDiff((0, 1, 2), (1, 1, 1), (2, 1, 2), acc=ACC)
        d5_dxdy2dz2 = fd.FinDiff((0, 1, 1), (1, 1, 2), (2, 1, 2), acc=ACC)
        d6_dx2dy2dz2 = fd.FinDiff((0, 1, 2), (1, 1, 2), (2, 1, 2), acc=ACC)
        
        self.f3D_deriv = []
        self.f3D_deriv.append(d_dx(self.f3D))
        self.f3D_deriv.append(d_dy(self.f3D))
        self.f3D_deriv.append(d_dz(self.f3D))
        f3D_xx = d2_dx2(self.f3D)
        f3D_xy = d2_dxdy(self.f3D)
        f3D_xz = d2_dxdz(self.f3D)
        f3D_yy = d2_dy2(self.f3D)
        f3D_yz = d2_dydz(self.f3D)
        f3D_zz = d2_dz2(self.f3D)
        f3D_xxy = d3_dx2dy(self.f3D)
        f3D_xxz = d3_dx2dz(self.f3D)
        f3D_xyy = d3_dxdy2(self.f3D)
        f3D_xyz = d3_dxdydz(self.f3D)
        f3D_yyz = d3_dy2dz(self.f3D)
        f3D_xzz = d3_dxdz2(self.f3D)
        f3D_yzz = d3_dydz2(self.f3D)
        f3D_xxyy = d4_dx2dy2(self.f3D)
        f3D_xxzz = d4_dx2dz2(self.f3D)
        f3D_yyzz = d4_dy2dz2(self.f3D)
        f3D_xxyz = d4_dx2dydz(self.f3D)
        f3D_xyyz = d4_dxdy2dz(self.f3D)
        f3D_xyzz = d4_dxdydz2(self.f3D)
        f3D_xxyyz = d5_dx2dy2dz(self.f3D)
        f3D_xxyzz = d5_dx2dydz2(self.f3D)
        f3D_xyyzz = d5_dxdy2dz2(self.f3D)
        f3D_xxyyzz = d6_dx2dy2dz2(self.f3D)
        
        fx = np.reshape(self.f3D_deriv[0], (nPosx * nPosy * nPosz, ), order='F')
        fy = np.reshape(self.f3D_deriv[1], (nPosx * nPosy * nPosz, ), order='F')
        fz = np.reshape(self.f3D_deriv[2], (nPosx * nPosy * nPosz, ), order='F')
        fxx = np.reshape(f3D_xx, (nPosx * nPosy * nPosz, ), order='F')
        fxy = np.reshape(f3D_xy, (nPosx * nPosy * nPosz, ), order='F')
        fxz = np.reshape(f3D_xz, (nPosx * nPosy * nPosz, ), order='F')
        fyy = np.reshape(f3D_yy, (nPosx * nPosy * nPosz, ), order='F')
        fyz = np.reshape(f3D_yz, (nPosx * nPosy * nPosz, ), order='F')
        fzz = np.reshape(f3D_zz, (nPosx * nPosy * nPosz, ), order='F')
        fxxy = np.reshape(f3D_xxy, (nPosx * nPosy * nPosz, ), order='F')
        fxxz = np.reshape(f3D_xxz, (nPosx * nPosy * nPosz, ), order='F')
        fxyy = np.reshape(f3D_xyy, (nPosx * nPosy * nPosz, ), order='F')
        fxyz = np.reshape(f3D_xyz, (nPosx * nPosy * nPosz, ), order='F')
        fyyz = np.reshape(f3D_yyz, (nPosx * nPosy * nPosz, ), order='F')
        fxzz = np.reshape(f3D_xzz, (nPosx * nPosy * nPosz, ), order='F')
        fyzz = np.reshape(f3D_yzz, (nPosx * nPosy * nPosz, ), order='F')
        fxxyy = np.reshape(f3D_xxyy, (nPosx * nPosy * nPosz, ), order='F')
        fxxzz = np.reshape(f3D_xxzz, (nPosx * nPosy * nPosz, ), order='F')
        fyyzz = np.reshape(f3D_yyzz, (nPosx * nPosy * nPosz, ), order='F')
        fxxyz = np.reshape(f3D_xxyz, (nPosx * nPosy * nPosz, ), order='F')
        fxyyz = np.reshape(f3D_xyyz, (nPosx * nPosy * nPosz, ), order='F')
        fxyzz = np.reshape(f3D_xyzz, (nPosx * nPosy * nPosz, ), order='F')
        fxxyyz = np.reshape(f3D_xxyyz, (nPosx * nPosy * nPosz, ), order='F')
        fxxyzz = np.reshape(f3D_xxyzz, (nPosx * nPosy * nPosz, ), order='F')
        fxyyzz = np.reshape(f3D_xyyzz, (nPosx * nPosy * nPosz, ), order='F')
        fxxyyzz = np.reshape(f3D_xxyyzz, (nPosx * nPosy * nPosz, ), order='F')

        self.Bn = np.array([self.inputfield[:, 3], fx, fy, fz,
                            fxx, fxy, fxz, fyy, fyz, fzz,
                            fxxy, fxxz, fxyy, fxyz, fyyz, fxzz, fyzz,
                            fxxyy, fxxzz, fyyzz, fxxyz, fxyyz, fxyzz,
                            fxxyyz, fxxyzz, fxyyzz,
                            fxxyyzz])
        # precalculate all coefficients
        # self.allCoeffs()

    def calcCoefficients(self, alphaindex):
        # Find interpolation coefficients for a cuboid
        realindex = self.basePointInds[alphaindex]
        # Find other vertices of current cuboid, and all neighbours in 6x6x6 neighbouring array
        inds = self.neighbourInd(realindex)
        # Alpha coefficients
        vals_unit_cube = self.Bn[:, inds]
        vals_unit_cube = np.reshape(vals_unit_cube, (len(inds) * 27, ), order='C')[:]
        # self.alphan[:, alphaindex] = np.dot(self.A, self.Bn[inds])
        self.alphan[:, alphaindex] = np.dot(self.A, vals_unit_cube)
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
