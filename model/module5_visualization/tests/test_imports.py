import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_m5 = os.path.dirname(_here)
sys.path.insert(0, _m5)

def test_import_trajectory_viz():
    import trajectory_viz
    assert True

def test_import_error_analysis():
    import error_analysis_viz
    assert True

def test_import_module1_viz():
    import module1_viz
    assert True

def test_import_baseline_comparison_viz():
    import baseline_comparison_viz
    assert True

def test_import_generate_report():
    import generate_report
    assert True
