import numpy as _np
import numpy.testing

import georges_core.geometry as ggeom


def test_projection():
    x1 = _np.arange(0, 11, 0.7)
    y1 = _np.sin(x1 / 2) / 2
    t1 = _np.cos(x1 / 2) / 4
    dist = _np.linalg.norm(_np.array((x1, y1)).T - _np.roll(_np.array((x1, y1)).T, -1, axis=0), axis=1)
    s1 = _np.concatenate((_np.array([0]), _np.cumsum(dist[0:-1])))

    x2 = _np.arange(0, 11, 0.3)
    y2 = _np.sin(x2)
    t2 = _np.cos(x2)
    dist = _np.linalg.norm(_np.array((x2, y2)).T - _np.roll(_np.array((x2, y2)).T, -1, axis=0), axis=1)
    s2 = _np.concatenate((_np.array([0]), _np.cumsum(dist[0:-1])))

    x3 = _np.arange(0, 11, 0.3)
    y3 = _np.cos(x3)
    t3 = -_np.sin(x3)
    dist = _np.linalg.norm(_np.array((x3, y3)).T - _np.roll(_np.array((x3, y3)).T, -1, axis=0), axis=1)
    s3 = _np.concatenate((_np.array([0]), _np.cumsum(dist[0:-1])))

    ref = _np.array([x1, y1, _np.zeros(len(x1)), t1, _np.zeros(len(x1)), s1]).T

    traj2 = _np.zeros([len(x2), 6])
    traj2[:, 0] = x2
    traj2[:, 1] = y2
    traj2[:, 3] = t2
    traj2[:, 5] = s2

    traj3 = _np.zeros([len(x3), 6])
    traj3[:, 0] = x3
    traj3[:, 1] = y3
    traj3[:, 3] = t3
    traj3[:, 5] = s3

    ref_traj = ggeom.ReferenceTrajectory(points=ggeom.Points(ref))
    interpolated_points = ggeom.project_on_reference(
        ref=ref_traj,
        trajectories=[
            ggeom.Trajectories(traj2),
            ggeom.Trajectories(traj3),
        ],
    )

    results_traj2_x = _np.array(
        [
            0.0,
            0.60504538,
            1.27861225,
            2.04370013,
            2.80712434,
            3.46408221,
            4.04657256,
            4.64547206,
            5.37074528,
            6.30716335,
            7.2351403,
            7.95161479,
            8.54735839,
            9.13152089,
            9.79374749,
            10.56058782,
        ],
    )
    results_traj2_y = _np.array(
        [
            0.0,
            0.5683203,
            0.9491912,
            0.88397243,
            0.32516188,
            -0.31350059,
            -0.77757098,
            -0.98656611,
            -0.78768929,
            0.02385135,
            0.81057411,
            0.98406235,
            0.76047144,
            0.2860057,
            -0.3572303,
            -0.90014222,
        ],
    )
    results_traj2_t = _np.array(
        [
            1.0,
            0.82190937,
            0.28594126,
            -0.4527417,
            -0.93475268,
            -0.93786057,
            -0.61078906,
            -0.06615062,
            0.60965911,
            0.99867367,
            0.57769539,
            -0.09639569,
            -0.63194807,
            -0.94668258,
            -0.9230137,
            -0.41874453,
        ],
    )

    results_traj3_x = _np.array(
        [
            _np.nan,
            0.53702304,
            1.4367595,
            2.22995118,
            2.86135295,
            3.4358608,
            4.0700778,
            4.8676595,
            5.76684973,
            6.54598023,
            7.18876488,
            7.77716304,
            8.3907603,
            9.08220873,
            9.81976162,
            10.50529182,
        ],
    )

    results_traj3_y = _np.array(
        [
            _np.nan,
            0.85262581,
            0.1322113,
            -0.60557889,
            -0.95028382,
            -0.94639485,
            -0.59232401,
            0.15301114,
            0.86266878,
            0.95916855,
            0.61611755,
            0.07623515,
            -0.510796,
            -0.93335267,
            -0.91473282,
            -0.47057661,
        ],
    )

    results_traj3_t = _np.array(
        [
            _np.nan,
            -0.50814747,
            -0.98369677,
            -0.78188324,
            -0.27341697,
            0.28671145,
            0.79197261,
            0.98029847,
            0.49023778,
            -0.258471,
            -0.78560748,
            -0.99393459,
            -0.85814975,
            -0.33329017,
            0.38178843,
            0.88148158,
        ],
    )

    numpy.testing.assert_allclose(interpolated_points[0, :, 0], results_traj2_x)
    numpy.testing.assert_allclose(interpolated_points[0, :, 1], results_traj2_y)
    numpy.testing.assert_allclose(interpolated_points[0, :, 3], results_traj2_t)
    numpy.testing.assert_allclose(interpolated_points[1, :, 0], results_traj3_x)
    numpy.testing.assert_allclose(interpolated_points[1, :, 1], results_traj3_y)
    numpy.testing.assert_allclose(interpolated_points[1, :, 3], results_traj3_t)
