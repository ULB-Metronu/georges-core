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
