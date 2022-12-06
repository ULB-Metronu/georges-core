*********
Particles
*********

Different particles are defined in `Georges_core`, such as Proton and Electron.
For each particle, theses properties are available:

.. list-table:: Particles implemented in `Georges_core`.
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Particle
     - Mass (MeV / c2)
     - Charge (C)
     - Lifetime
     - Gyromagnetic factor
   * - Proton
     - 938.27203
     - 1.60217649e-19
     - N.A
     - 1.7928473505
   * - Electron
     - 0.510998946
     - -1.60217649e-19
     - N.A
     - -2.0011596521810997

.. jupyter-execute::
    :hide-output:

    import georges_core.particles

.. jupyter-execute::

    p = georges_core.particles.Proton
    p

.. jupyter-execute::

    e = georges_core.particles.Electron
    e

API
---

.. automodule:: georges_core.particles
   :members:
   :undoc-members:
   :show-inheritance:
