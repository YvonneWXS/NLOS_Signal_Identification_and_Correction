# module3_adaptive/detector.py -- Wrapper delegating to shift_detector
# ponytail: delegates to original model_2 implementation
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from shift_detector import *
