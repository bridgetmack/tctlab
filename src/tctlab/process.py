'''These functions are used to analyze raw data from TCT output. An input datalocation is needed to run'''

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
import mplhep as hep
hep.style.use("LHCb")

import sys
datalocation = sys.argv[1]
sys.path.insert(0, datalocation)

