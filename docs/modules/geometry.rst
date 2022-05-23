********
Geometry
********

The code presented in this section is a Python3 re-implementation of the methods described in the book of D.Sunday
`Pratical Geometry Algorithms <https://geomalgorithms.com>`_. It implements class that represent and compute operations
on vectors, lines, points and rays.


.. jupyter-execute::
    :hide-output:

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as _np
    import georges_core
    import georges_core.geometry as ggeom

.. jupyter-execute::
    :hide-output:

    x1 = _np.linspace(0, 20, 50)
    y1 = _np.sin(x1)

    x2 = _np.linspace(0, 20, 75)
    y2 = _np.sin(0.5*x2)

    x3 = _np.linspace(0, 20, 100)
    y3 = _np.sin(3*x3)

.. jupyter-execute::
    :hide-output:

    ref = _np.array([x1, y1, _np.zeros(len(x1)), _np.gradient(y1, x1[1]-x1[0]), _np.zeros(len(x1))]).T

    traj2 = _np.zeros([len(x2), 5])
    traj2[:,0] = x2
    traj2[:,1] = y2
    traj2[:,4] = _np.gradient(y2, x2[1]-x2[0])

    traj3 = _np.zeros([len(x3), 5])
    traj3[:,0] = x3
    traj3[:,1] = y3
    traj3[:,4] = _np.gradient(y3, x3[1]-x3[0])

.. jupyter-execute::
    :hide-output:

    ref_traj = ggeom.ReferenceTrajectory(points=ggeom.Points(ref))
    interpolated_points = ggeom.project_on_reference(ref=ref_traj, trajectories = [ggeom.Trajectories(traj2),
                                                                                   ggeom.Trajectories(traj3)])

.. jupyter-execute::

    fig, (ax1, ax2) = plt.subplots(figsize=(20, 8), nrows=2, sharex=True)

    ax1.plot(ref_traj.data[:, 0], ref_traj.data[:, 1], '-b*')
    ax1.plot(x2, y2, ls='-', color='r', marker='*')
    ax1.plot(interpolated_points[0,:, 0], interpolated_points[0,:, 1], 'ok', fillstyle='none')

    for k in list(map(ggeom.Plane, ref_traj.frenet_frames().data)):

        ax1.plot([k.data[0][0] + 1.5*k.data[1][1], k.data[0][0] - 1.5*k.data[1][1]],
                [k.data[0][1] - 1.5*k.data[1][0], k.data[0][1] + 1.5*k.data[1][0]],
                ls='--',
                lw=0.5,
                color='k'
                )

    ax1.set_aspect('equal')
    ax1.set_ylabel('y')

    ax2.plot(ref_traj.data[:, 0], ref_traj.data[:, 1], '-b*')
    ax2.plot(x3, y3, ls='-', color='darkorange', marker='*')
    ax2.plot(interpolated_points[1, :, 0], interpolated_points[1, :, 1], 'ok', fillstyle='none')

    for k in list(map(ggeom.Plane, ref_traj.frenet_frames().data)):

        ax2.plot([k.data[0][0] + 1.5*k.data[1][1], k.data[0][0] - 1.5*k.data[1][1]],
                [k.data[0][1] - 1.5*k.data[1][0], k.data[0][1] + 1.5*k.data[1][0]],
                ls='--',
                lw=0.5,
                color='k',
                )

    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')