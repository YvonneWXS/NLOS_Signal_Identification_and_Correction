# module1_nlos/features.py -- Delegates to NodeFeature_Generate from model_2
# ponytail: original GAT_V2026.py not yet split; this is the plan-compliant entry point
import sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
# Import original module for backward compatibility
exec('from NodeFeature_Generate import *')
