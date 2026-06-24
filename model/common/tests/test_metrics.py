import sys, numpy as np, numpy as np
sys.path.insert(0, r'D:\3_document\4_research\NLOS Signal Identification and Correction\model')
from common.metrics import cep50, cep95, rmse, all_metrics

def test_cep50():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(cep50(errors) - 3.0) < 0.01

def test_cep95():
    errors = np.linspace(0, 100, 101)
    c95 = cep95(errors)
    assert abs(c95 - 95.0) < 1.0

def test_rmse():
    errors = np.array([3.0, 4.0])
    assert abs(rmse(errors) - np.sqrt(12.5)) < 0.01

def test_all_metrics():
    errors = np.array([1.0, 2.0, 3.0])
    result = all_metrics(errors)
    assert 'cep50' in result
    assert 'rmse' in result
    assert 'mean' in result

def test_empty_array():
    errors = np.array([])
    result = all_metrics(errors)
    assert result == {} or len(result) == 0
