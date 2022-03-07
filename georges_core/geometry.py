"""
TODO
"""
from typing import Optional
import numpy as _np
import quaternion


class Intersections:
    def __init__(self, lines1, line2):
        self._lines1, self._line2 = self.validate_format(lines1, line2)
        self._u = None
        self._v = None
        self._w = None
        self._parallels = None
        self._coincidents = None
        self._s_intersections = None
        self._t_intersections = None

    def validate_format(self, lines1, line2):
        assert lines1.ndim > 1
        lines1 = _np.atleast_3d(lines1)
        assert line2.shape == (2, 2)
        return lines1, line2

    @property
    def u(self):
        if self._u is None:
            self._u = self._lines1[:, 1] - self._lines1[:, 0]
        return self._u

    @property
    def v(self):
        if self._v is None:
            self._v = self._line2[1] - self._line2[0]
        return self._v

    @property
    def w(self):
        if self._w is None:
            self._w = self._lines1[:, 0] - self._line2[0]
        return self._w

    @property
    def parallels(self):
        if self._parallels is None:
            self._parallels = _np.cross(self.u, self.v) == 0.0
        return self._parallels

    @property
    def not_parallels(self):
        return _np.invert(self.parallels)

    @property
    def coincidents(self):
        if self._coincidents is None:
            self._coincidents = self.parallels & (_np.cross(self.w, self.v) == 0.0)
        return self._coincidents

    @property
    def not_coincidents(self):
        return _np.invert(self.coincidents)

    @property
    def parameters_at_intersections(self):
        if self._s_intersections is None:
            s_num = _np.cross(-self.v, self.w)
            denom = _np.ma.masked_values(_np.cross(self.v, self.u), 0.0)
            self._s_intersections = s_num / denom
        if self._t_intersections is None:
            t_num = _np.cross(self.u, self.w)
            self._t_intersections = -t_num / denom
        return (
            self._s_intersections,
            self._t_intersections
        )

    @property
    def intersects_ray_segment(self):
        s_intersections, t_intersections = self.parameters_at_intersections
        return ((s_intersections >= 0) & (t_intersections >= 0) & (t_intersections <= 1)).filled(False)

    @property
    def intersections(self):
        s_intersections, t_intersections = self.parameters_at_intersections
        return _np.ma.masked_where(
            ~_np.column_stack((self.intersects_ray_segment, self.intersects_ray_segment)),
            self._lines1[:, 0] + _np.stack([self._s_intersections, self._s_intersections]).T * self._u
        )


class PrimitivesType(type):
    pass


class Primitives(metaclass=PrimitivesType):
    def __init__(self, data: _np.ndarray):
        self._data: Optional[_np.ndarray] = data

    @property
    def data(self):
        return self._data

    def _validate(self):
        assert isinstance(self._data, _np.ndarray)

    @property
    def first(self):
        """

        Returns:

        """
        return self._data[0]


class Points(Primitives):
    def __init__(self, points: _np.ndarray):
        """

        Args:
            points:
        """
        super().__init__(points)
        self._validate()

    def _validate(self):
        super()._validate()
        assert self._data.ndim == 2
        assert self._data.shape[0] >= 1
        assert self._data.shape[1] >= 3

    def number_points(self):
        return self._data.shape[0]

    # checker ordre des coordonnées: pour l'instant (x,y,z,p,t,s);
    @property
    def x(self):
        return self._data[:, 0]

    @property
    def y(self):
        return self._data[:, 1]

    @property
    def z(self):
        return self._data[:, 2]

    @property
    def t(self):
        return self._data[:, 3]

    @property
    def p(self):
        return self._data[:, 4]

    @property
    def s(self):
        return self._data[:, 5]

    @property
    def coordinates(self):
        return self._data[:, 0:3]


class Point(Points):
    def _validate(self):
        super()._validate()
        assert self.data.ndim == 2
        assert self.data.shape[0] == 1


class Vector(Point):
    # TODO : add possibility to define a Vector with 2 Points or a Segment
    pass


class Vectors(Points):
    pass


class Lines(Primitives):
    def __init__(self, lines: _np.ndarray):
        super().__init__(lines)
        self._u = None

    def _validate(self):
        super()._validate()
        assert self.data.ndim == 3
        assert self.data.shape[0] > 1

    @property
    def p0(self):
        return self._data[:, 0]

    @property
    def p1(self):
        return self._data[:, 1]

    @property
    def u(self):
        if self._u is None:
            self._u = self._data[:, 1] - self._data[:, 0]
        return self._u


class Line(Lines):
    def _validate(self):
        super()._validate()
        assert self.data.ndim == 3
        assert self.data.shape[0] == 1


class Trajectories(Lines):
    pass


class Rays(Lines):
    pass


class Ray(Rays):
    pass


class Segments(Rays):
    pass


class Segment(Segments):
    pass


class Planes(Primitives):
    def __init__(self, plane: _np.ndarray):
        super().__init__(plane)

    def _validate(self):
        super()._validate()
        assert self.data.ndim == 3
        assert self.data.shape[1:] == (2, 3)


class Plane(Primitives):
    def __init__(self, plane: _np.ndarray):
        super().__init__(plane)

    def _validate(self):
        super()._validate()
        assert self.data.ndim == 2
        assert self.data.shape == (2, 3)

    @property
    def point(self):
        return self._data[0]

    @property
    def normal(self):
        return self._data[1]


class ReferenceTrajectory:

    def __init__(self, points: Points):
        """Initialized with an array of 6D data points (x, y, z, t, p, s)"""
        self._points = points
        self._data = points.data

    def frenet_frames(self):
        """Returns an array of frenet frames (representation of planes)"""
        points = self._points
        tangent_vector = construct_tangent_vectors(points)
        for pt, point in enumerate(points.data):
            plan = _np.concatenate((Point(_np.array([point])).coordinates, _np.array([tangent_vector.data[pt]])))
            if pt == 0:
                planes = plan.reshape((1, 2, 3))
            else:
                planes = _np.vstack((planes, plan.reshape((1, 2, 3))))
        return Planes(planes)

    @property
    def data(self):
        return self._data


def construct_tangent_vectors(p: Points,
                              norm: float = 1.0) -> Vectors:
    """
    Constructs the tangent vector to the trajectory at the point p
    """
    _ = _np.zeros((p.data.shape[0], 3))
    _[:, 2] = -p.t
    q1 = quaternion.from_rotation_vector(_)
    _[:, 2] = 0
    _[:, 1] = p.p
    q2 = quaternion.from_rotation_vector(_)
    q = q2 * q1
    end_points = _np.matmul(_np.linalg.inv(quaternion.as_rotation_matrix(q)), _np.array([norm, 0.0, 0.0]))
    return Vectors(end_points)


def build_segments(trajectory: Points):
    """Builds a set of NT-1 segments from a list of NT points. Uses the spacial coordinates x, y, z to build
    a segment in space, and also associate the start and end points for the other coordiantes (t, p and s)."""
    for pt in range(trajectory.data.shape[0] - 1):
        segment = trajectory.data[pt:pt + 2]
        if pt == 0:
            segments = segment.reshape(1, 2, 6)
        else:
            segments = _np.vstack((segments, segment.reshape(1, 2, 6)))
    return Segments(segments)


def intersection_segment_plane(segment: Segment, plane: Plane):
    """Provides the intersection between a plane and a segment. Returns -1 if negative."""
    # Attention, un segment isolé doit quand même avoir une dimension "vertiale" --> ndim = (1, 2 , 6)
    u = segment.u[0, 0:3]
    w = segment.p0[0, 0:3] - plane.point
    D = _np.dot(plane.normal, u)
    N = -_np.dot(plane.normal, w)

    if _np.abs(D) < 1e-4:
        if N == 0:
            return 2  # Segment dans le plan
        else:
            return -1  # No intersection
    sI = N / D
    if sI < 0 or sI > 1:
        return -1
    return sI


def intersection_point(segment: Segment, sI: float):
    """
    Compute the intersection point between a segment and a plane, given the value sI, and interpolate all coordinates
    """
    pm = segment.p0 + segment.u * sI
    return pm


def project_on_reference(ref: ReferenceTrajectory, trajectories: list):
    """
    Projects a set of NT trajectories on a reference trajectory (array of points of length NR).
    For each trajectory (array of points of length NI), computes the coordinates associated with each
    data point of the reference trajectory (so it returns an array of NT projected trajectories, each of
    length NR).
    """
    results = _np.zeros((len(trajectories), ref.data.shape[0], 6))
    for it, t in enumerate(trajectories):
        for ir, r in enumerate(ref.frenet_frames().data):
            for s in build_segments(Points(t.data)).data:
                segment = Segment(s.reshape(1, 2, 6))
                i = intersection_segment_plane(segment, Plane(r))
                if i < 0:  # we don't have an intersection, try the next one
                    continue
                # compute the intersection point (x, y, z) and interpolate (t, p and s) using i
                i_coordinates = intersection_point(segment,
                                                   i)
                results[it, ir] = i_coordinates
                break
    return results
