import importlib

import SVD_analysis.delete_SVDprobesfull
importlib.reload(SVD_analysis.delete_SVDprobesfull)
from SVD_analysis.delete_SVDprobesfull import delete_allprobes_full

try:
    delete_allprobes_full()
except Exception as e:
    raise(e)