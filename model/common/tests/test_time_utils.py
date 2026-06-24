import sys
sys.path.insert(0, r'D:\3_document\4_research\NLOS Signal Identification and Correction\model')
from common.time_utils import gps_to_datetime, datetime_to_gps
from datetime import datetime

def test_roundtrip():
    dt = datetime(2016, 6, 6, 12, 0, 0)
    week, sec = datetime_to_gps(dt)
    dt2 = gps_to_datetime(week, sec)
    assert abs((dt - dt2).total_seconds()) < 1.0

def test_gps_epoch():
    dt = datetime(1980, 1, 6, 0, 0, 0)
    week, sec = datetime_to_gps(dt)
    assert week == 0
    assert sec == 0.0
