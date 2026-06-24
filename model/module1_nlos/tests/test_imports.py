# Smoke tests for Module 1
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_m1 = os.path.dirname(_here)
_model2 = r'D:\\3_document\\4_research\\NLOS Signal Identification and Correction\\model_2\\part1_GAT\\model'
_model2_radio = r'D:\\3_document\\4_research\\NLOS Signal Identification and Correction\\model_2\\part1_GAT\\RadioGAT-Multi-band-Radiomap-Reconstruction'
for p in [_m1, os.path.join(os.path.dirname(_m1), 'common'), _model2, _model2_radio]:
    if os.path.isdir(p): sys.path.insert(0, p)

def test_import_data_loader():
    import data_loader
    assert True

def test_import_features():
    import features
    assert True

def test_import_model():
    import model
    assert True

def test_import_loss():
    import loss
    assert True

def test_config_yaml_exists():
    cfg_path = os.path.join(_m1, 'config.yaml')
    assert os.path.exists(cfg_path)
