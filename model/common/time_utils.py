# common/time_utils.py -- GPS time conversion utilities
from datetime import datetime

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

def gps_to_datetime(gps_week, gps_seconds):
    total_sec = gps_week * 7 * 86400 + gps_seconds
    from datetime import timedelta
    return GPS_EPOCH + timedelta(seconds=total_sec)

def datetime_to_gps(dt):
    delta = dt - GPS_EPOCH
    total_sec = delta.total_seconds()
    week = int(total_sec // (7 * 86400))
    sec = total_sec % (7 * 86400)
    return week, sec
