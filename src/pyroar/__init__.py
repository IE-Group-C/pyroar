
import logging
DEFAULT_PARTITIONS = 1
logging.basicConfig(level=logging.INFO, format="%(message)s")

__version__ = "0.0.1"

from .pyroar import Pyroar
from .manipulator import Manipulator

