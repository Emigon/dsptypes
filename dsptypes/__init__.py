import warnings

from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
set_application_registry(ureg)

ureg.define('samples = 1') # for unspecified units

Q_ = ureg.Quantity

with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  Q_([])

from .parametric import *
from .signal1D import *
