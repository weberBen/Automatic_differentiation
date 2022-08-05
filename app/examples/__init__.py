import sys
import os
from pathlib import Path

parent_dir = Path(sys.path[0]).parent.absolute()

sys.path.insert(0, '%s%ssrc' % (parent_dir, os.sep))
