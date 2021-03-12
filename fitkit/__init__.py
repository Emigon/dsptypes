import os
os.environ['PINT_ARRAY_PROTOCOL_FALLBACK'] = "0" # for numpy support with pint

from .parametric import *
from .gui import Gui
