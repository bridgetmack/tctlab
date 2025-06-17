from ctypes import *
import os, sys, pyvisa, datetime
if sys.version_info >= (3, 0):
    import urllib.parse
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar

import libximc.highlevel as ximc

# cur_dir = os.path.abspath(os.path.dirname(__file__)) #makes variable that lists current directory
# ximc_dir = os.path.join(cur_dir, "..", "..", "..", "ximc") #adds directory that ximc is in
# ximc_package_dir = os.path.join(ximc_dir, "crossplatform", "wrappers", "python") #gets us into the directory that the python library is in
# sys.path.append(ximc_dir) #adds library to python path

# #add DLLs to path
# arch_dir = os.path.join(cur_dir, "ximc", "win64")
# # libdir = os.path.join(ximc_dir, arch_dir)
# os.add_dll_directory(arch_dir)
# os.add_dll_directory(cur_dir)
# try: 
#     from pyximc import *
# except ImportError as err:
#     print("Cannot import pyximc module.")
#     exit
# except OSError as err:
#     if err.winerror == 193:
#         print("Bit depth does not correspond with operating system")
#     elif err.winerror == 126:
#         print("One of the library files is missing")
#     else:
#         print(err)

# sbuf= create_string_buffer(64)
# lib.ximc_version(sbuf)

def test_library():
    print("Library version: ", ximc.ximc_version())

