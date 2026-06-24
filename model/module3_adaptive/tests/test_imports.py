import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_m3 = os.path.dirname(_here)
_model_root = os.path.dirname(_m3)
_fusion = r'D:\3_document\4_research\NLOS Signal Identification and Correction\model_2\part2_FactorGraphLocalizationFusion\model'
for p in [_model_root, _fusion]:
    if os.path.isdir(p): sys.path.insert(0, p)

def test_import_tracker():
    from module3_adaptive import tracker
    assert True

def test_import_detector():
    from module3_adaptive import detector
    assert True

def test_import_selector():
    from module3_adaptive import selector
    assert True
