import sys, os
sys.path.insert(0, r'D:\3_document\4_research\NLOS Signal Identification and Correction\model')
from common.sp3_reader import SP3Reader

SP3_PATH = r'D:\3_document\4_research\NLOS Signal Identification and Correction\data\dataset\berlin1_potsdamer_platz\gbm19001.sp3'

def test_sp3_parse():
    reader = SP3Reader(SP3_PATH)
    stats = reader.get_statistics()
    assert stats['total_epochs'] > 0
    assert stats['total_satellites'] > 0

def test_sp3_has_satellite():
    reader = SP3Reader(SP3_PATH)
    assert reader.has_satellite('G12')

def test_sp3_position():
    reader = SP3Reader(SP3_PATH)
    pos = reader.get_satellite_position(1900, 126641.5, 'G12')
    assert pos is not None
    assert len(pos) == 3
    # Position should be in meters, magnitude ~26,000 km
    mag = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5 / 1000.0
    assert 20000 < mag < 30000

def test_sp3_nonexistent():
    reader = SP3Reader(SP3_PATH)
    assert not reader.has_satellite('XX99')
    pos = reader.get_satellite_position(1900, 126641.5, 'XX99')
    assert pos is None
