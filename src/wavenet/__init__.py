import os
import sys

path_self = os.path.dirname(os.path.realpath(__file__))
path_core = os.path.join(path_self, "..", "..", "base")
print("Extending python path with {}".format(os.path.normpath(path_core)))
sys.path.insert(0, path_core)