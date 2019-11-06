from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
set_application_registry(ureg)

Q_ = ureg.Quantity

from .parametric import *
