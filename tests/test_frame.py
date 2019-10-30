import numpy as np
from georges_core import ureg as _
from georges_core import Frame, FrameFrenet


def test_frame_frenet_does_only_z_rotation():
    f1 = Frame()
    f2 = Frame(f1)
    ff1 = FrameFrenet(f2)
    ff1.rotate_y(-2 * _.radian)
    assert np.all(np.isclose(ff1.get_rotation_vector(), np.array([0, 0, 0])))

    ff1.rotate_z(-2 * _.radian)
    assert np.all(np.isclose(ff1.get_rotation_vector(), np.array([0, 0, 0])))

    ff1.rotate_x(3 * _.radian)
    assert np.all(np.isclose(ff1.get_rotation_vector(), np.array([3, 0, 0])))
