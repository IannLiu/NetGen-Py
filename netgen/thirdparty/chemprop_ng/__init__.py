import os
import sys

curr_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(curr_path)
sys.path.append(os.path.join(curr_path, 'chemprop'))

import chemprop

