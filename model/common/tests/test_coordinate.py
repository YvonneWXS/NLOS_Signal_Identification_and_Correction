import sys, numpy as np
sys.path.insert(0, r'D:\3_document\4_research\NLOS Signal Identification and Correction\model')
from common.coordinate import lla_to_ecef, ecef_to_lla, ecef_to_enu, compute_azimuth_elevation

def test_lla_ecef_roundtrip():
    lat, lon, h = 52.5, 13.4, 100.0
    ecef = lla_to_ecef(lat, lon, h)
    assert ecef.shape == (3,)
    lat2, lon2, h2 = ecef_to_lla(ecef[0], ecef[1], ecef[2])
    assert abs(lat - lat2) < 0.001
    assert abs(lon - lon2) < 0.001
    assert abs(h - h2) < 1.0

def test_ecef_to_enu():
    ref = lla_to_ecef(52.5, 13.4, 100.0)
    tgt = lla_to_ecef(52.51, 13.41, 100.0)
    enu = ecef_to_enu(ref, tgt)
    assert enu.shape == (3,)

def test_ecef_to_enu_batch():
    ref = lla_to_ecef(52.5, 13.4, 100.0)
    tgts = np.array([lla_to_ecef(52.51, 13.41, 100.0), lla_to_ecef(52.49, 13.39, 100.0)])
    enu = ecef_to_enu(ref, tgts)
    assert enu.shape == (2, 3)

def test_azimuth_elevation():
    rx = lla_to_ecef(52.5, 13.4, 100.0)
    sv = lla_to_ecef(52.5, 13.4, 20200000.0)  # directly overhead
    az, el = compute_azimuth_elevation(rx, sv)
    assert el > 80.0  # nearly overhead
