*****************
Kinematics module
*****************

This module is a simple standalone helper module providing various relativistic conversion functions
as well as range computations for protons in water (using the IEC60601 range in water conversion).
Using `Pint <https://pint.readthedocs.io/en/stable/>`_, no conversion is needed. To use the unit registry,
simply import the module units:

.. jupyter-execute::
    :hide-output:

    import georges_core
    from georges_core.units import ureg as _ureg

The module is capable of determining the type of the quantity that you pass to the class.
- Type of particles
- Kinetic True / False

.. jupyter-execute::

    kin = georges_core.Kinematics(230 * _ureg.MeV, particle=georges_core.particles.Proton, kinetic=True)
    kin

.. jupyter-execute::

    kin = georges_core.Kinematics(1230 * _ureg.MeV, particle=georges_core.particles.Proton, kinetic=False)
    kin

.. jupyter-execute::

    kin = georges_core.Kinematics(15 * _ureg.MeV, particle=georges_core.particles.Electron, kinetic=True)
    kin