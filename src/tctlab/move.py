from ctypes import *
import os, sys, pyvisa, datetime
if sys.version_info >= (3, 0):
    import urllib.parse
import matplotlib.pyplot as plt
import numpy as np
from progress.bar import Bar

import libximc.highlevel as ximc

z_limit = 39000

def test_library():
    print("Library version: ", ximc.ximc_version())

probe_flags = ximc.EnumerateFlags.ENUMERATE_ALL_COM
enum_hits = "addr="
dev_enum= ximc.enumerate_devices(probe_flags, enum_hits)

def test_enum_handles():
    '''prints device enumeration handles'''
    print("Device Enumeration Handles: ")
    for i in range(1, len(dev_enum)):
        print(dev_enum[i]["uri"])

'''If this is your first time with the motors plugged into this machine, you need to check and make sure that the axes are defined correctly with this scheme.'''

try:
    motor_x = ximc.Axis(dev_enum[1]["uri"])
    motor_y = ximc.Axis(dev_enum[4]["uri"])
    motor_z = ximc.Axis(dev_enum[3]["uri"])
except: 
    print("Make sure motors are plugged in properly")

motor_x.open_device()
motor_y.open_device()
motor_z.open_device()



motor_x.command_move(0, 0)
motor_y.command_move(0, 0)
motor_z.command_move(17000, 0)

## add make header to this, so that the header gets made as soon as the scans start?
