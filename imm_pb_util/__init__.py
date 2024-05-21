# Setup import path
import sys
from pathlib import Path
BASEDIR = Path(__file__).parent
if str(BASEDIR) not in sys.path:
    sys.path.insert(0, str(BASEDIR))