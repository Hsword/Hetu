import os
import sys
for p in ['site_package', 'build/lib']:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), p))