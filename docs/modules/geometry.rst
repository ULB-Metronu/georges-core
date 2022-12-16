********
Geometry
********

The code presented in this section is a Python3 re-implementation of the methods described in the book of D.Sunday
`Pratical Geometry Algorithms <https://geomalgorithms.com>`_. It implements class that represent and compute operations
on vectors, lines, points and rays.

This algorithm is used in `Zgoubidoo` where we must “re-align” the tracks on a reference one to compute Twiss parameters.
The main function in this module is::

    project_on_reference(ref: ReferenceTrajectory, trajectories: list):

This method takes as input a Reference trajectory and tracks to interpolate and
returns the new coordinates for the tracks.
A Reference trajectory is a class that take as input a `numpy.ndarray` where
columns are x,y,z,t,p and s.

The example below shows an example where, at each point of the reference tracks
(in blue, we find the perpendicular points on given tracks (in red and dark orange).

.. jupyter-execute::
    :hide-output:

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as _np
    import georges_core
    import georges_core.geometry as ggeom

.. jupyter-execute::
    :hide-output:

    x1 = _np.arange(0,11, 0.7)
    y1 = _np.sin(x1/2)/2
    t1 = _np.cos(x1/2)/4
    dist = _np.linalg.norm(_np.array((x1, y1)).T - _np.roll(_np.array((x1, y1)).T, -1, axis=0),axis=1)
    s1 = _np.concatenate((_np.array([0]), _np.cumsum(dist[0:-1])))

    x2 =  _np.arange(0,11, 0.3)
    y2 = _np.sin(x2)
    t2 = _np.cos(x2)
    dist = _np.linalg.norm(_np.array((x2, y2)).T - _np.roll(_np.array((x2, y2)).T, -1, axis=0),axis=1)
    s2 = _np.concatenate((_np.array([0]), _np.cumsum(dist[0:-1])))


    x3 = _np.arange(0,11, 0.3)
    y3 = _np.cos(x3)
    t3 = -_np.sin(x3)
    dist = _np.linalg.norm(_np.array((x3, y3)).T - _np.roll(_np.array((x3, y3)).T, -1, axis=0),axis=1)
    s3 = _np.concatenate((_np.array([0]), _np.cumsum(dist[0:-1])))

.. jupyter-execute::
    :hide-output:

    ref = _np.array([x1, y1, _np.zeros(len(x1)), t1, _np.zeros(len(x1)), s1]).T

    traj2 = _np.zeros([len(x2), 6])
    traj2[:,0] = x2
    traj2[:,1] = y2
    traj2[:,3] = t2
    traj2[:,5] = s2

    traj3 = _np.zeros([len(x3), 6])
    traj3[:,0] = x3
    traj3[:,1] = y3
    traj3[:,3] = t3
    traj3[:,5] = s3

.. jupyter-execute::
    :hide-output:

    ref_traj = ggeom.ReferenceTrajectory(points=ggeom.Points(ref))
    interpolated_points = ggeom.project_on_reference(ref=ref_traj, trajectories = [ggeom.Trajectories(traj2),
                                                                               ggeom.Trajectories(traj3)])
.. jupyter-execute::
    :hide-output:

    def get_lines_points(point, normal):

        if normal[1] == 0:
            return [[point[0], point[0]],[point[1]-1, point[1]+1]]

        else:

            xa = point[0] - 0.5
            xb = point[0] + 0.5

            ya = point[1]-((normal[0]/normal[1])*(xa - point[0]))
            yb = point[1]-((normal[0]/normal[1])*(xb - point[0]))

            return [[xa,xb],[ya,yb]]

.. jupyter-execute::

    fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=1, sharex=False)

    axs[0].plot(ref_traj.data[:, 0], ref_traj.data[:, 1], '-b*')
    axs[0].plot(x2, y2, ls='-', color='r', marker='x')
    axs[0].plot(interpolated_points[0,:, 0], interpolated_points[0,:, 1], 'ok', fillstyle='none')

    for k in list(map(ggeom.Plane, ref_traj.frenet_frames().data)):
        xl, yl = get_lines_points(k.point, k.normal)
        axs[0].plot(xl, yl, ls='--', lw=0.5, color='k')

    axs[0].set_xticks(_np.arange(0, 11+1, 1.0))
    axs[0].set_yticks(_np.arange(-3, 4, 1.0))
    axs[0].axis('equal')
    axs[0].set_ylabel('y')
    axs[0].set_ylim([-1.5,1.5])
    axs[0].grid(True)

    axs[1].plot(ref_traj.data[:, 0], ref_traj.data[:, 1], '-b*')
    axs[1].plot(x3, y3, ls='-', color='darkorange', marker='x')
    axs[1].plot(interpolated_points[1,:, 0], interpolated_points[1,:, 1], 'ok', fillstyle='none')

    for k in list(map(ggeom.Plane, ref_traj.frenet_frames().data)):
        xl, yl = get_lines_points(k.point, k.normal)
        axs[1].plot(xl, yl, ls='--', lw=0.5, color='k')

    axs[1].set_xticks(_np.arange(0, 11+1, 1.0))
    axs[1].set_yticks(_np.arange(-3, 4, 1.0))
    axs[1].axis('equal')
    axs[1].set_ylabel('y')
    axs[1].set_ylim([-1.5,1.5])
    axs[1].grid(True)
