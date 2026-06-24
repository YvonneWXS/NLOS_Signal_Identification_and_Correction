# module3_adaptive/selector.py -- Wrapper delegating to run_module3
# ponytail: delegates to original model_2 implementation
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from run_module3 import *
