import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'pytc.log'

# Configure logging
def setup_logging(level=logging.INFO):
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get 'pytc' logger
    logger = logging.getLogger('pytc')
    logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid double logging if root is also configured
    logger.propagate = False

# Initialize logging when package is imported
setup_logging()

# Primary API - JAX autodiff implementations
from . import tc
from . import xtc
from . import scf
from . import df
from . import kmat
from . import lmat
from . import tc_helper

# Submodules
from . import ansatz
from . import jastrow
from . import vmc

__all__ = [
    'tc',
    'xtc', 
    'scf',
    'df',
    'kmat',
    'lmat',
    'tc_helper',
    'ansatz',
    'jastrow',
    'vmc',
]
