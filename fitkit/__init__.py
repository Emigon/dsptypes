import os
os.environ['PINT_ARRAY_PROTOCOL_FALLBACK'] = "0" # for numpy support with pint

from .signal1D import *
from .parametric import *
