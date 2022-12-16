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

The module is capable of determining the type of the quantity that you pass to
the class. The syntax is as follow::

    georges_core.Kinematics(quantity, particle, kinetic)

* Quantity
    * Total Energy
    * Kinetic Energy
    * Momentum
    * Relativist Beta
    * Relativist Gamma
    * Magnetic rigidity
    * Range: Only for protons
    * pv
* Type of particles
    * Protons
    * Electrons

.. note::

    The energy is the total energy by default. To have the kinetic energy instead, the argument
    `kinetic = True` must be passed to the constructor. If the energy is below the mass energy
    of the particles, the energy is the kinetic energy.

.. jupyter-execute::

    kin = georges_core.Kinematics(230 * _ureg.MeV, particle=georges_core.particles.Proton, kinetic=True)
    kin

.. jupyter-execute::

    kin = georges_core.Kinematics(1230 * _ureg.MeV, particle=georges_core.particles.Proton, kinetic=False)
    kin

.. jupyter-execute::

    kin = georges_core.Kinematics(15 * _ureg.MeV, particle=georges_core.particles.Electron, kinetic=True)
    kin
