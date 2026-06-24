import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import_baseline_runner():
    import baseline_runner
    assert True

def test_import_statistical_test():
    import statistical_test
    assert True

def test_import_results_aggregator():
    import results_aggregator
    assert True
