# Smoke test: factory registration and solver interface
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_factory_lists_methods():
    import module2_localization.standard_ls
    import module2_localization.wls
    import module2_localization.hard_threshold
    import module2_localization.factor_graph
    import module2_localization.raim
    import module2_localization.irls
    import module2_localization.kalman
    import module2_localization.cno_weighted
    import module2_localization.snr_weighted
    import module2_localization.dnn
    import module2_localization.gat_e2e
    import module2_localization.ins_gnss
    from module2_localization.factory import LocalizationFactory
    methods = LocalizationFactory.list_methods()
    assert len(methods) >= 10
    assert 'standard_ls' in methods
    assert 'ekf' in methods

def test_solver_returns_correct_shape():
    from module2_localization.factory import LocalizationFactory
    solver = LocalizationFactory.create('standard_ls')
    obs = np.array([20000.0, 21000.0, 22000.0, 23000.0])
    svp = np.array([[15000,5000,20000],[16000,6000,21000],[14000,4000,19000],[17000,7000,22000]])
    pos, clk, details = solver.solve(obs, svp)
    assert pos.shape == (3,)
    assert isinstance(clk, float)
    assert 'converged' in details

def test_snr_weighted_uses_cno_fallback():
    from module2_localization.factory import LocalizationFactory
    solver = LocalizationFactory.create('snr_weighted')
    obs = np.array([20000.0, 21000.0, 22000.0, 23000.0])
    svp = np.array([[15000,5000,20000],[16000,6000,21000],[14000,4000,19000],[17000,7000,22000]])
    info = {'cno': np.array([40.0, 35.0, 30.0, 45.0])}
    pos, clk, details = solver.solve(obs, svp, additional_info=info)
    assert pos.shape == (3,)
