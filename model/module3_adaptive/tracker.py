# module3_adaptive/tracker.py -- Wrapper delegating to residual_feedback
# ponytail: delegates to original model_2 implementation
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from residual_feedback import *
