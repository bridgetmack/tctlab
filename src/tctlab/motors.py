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
    print("Device Enumeration Handle: ", repr(dev_enum))
    print("Device Enumeration Handle Type: ", repr(type(dev_enum)))

# dev_count = ximc.get_device_count(dev_enum)

# def test_dev_count():
#     '''prints number of devices library finds'''
#     print('Device Count: ' + repr(dev_count))

controller_name= ximc.controller_name_t()
x= ximc.get_device_name(dev_enum, 1) ## COM 3
y= ximc.get_device_name(dev_enum, 2) ## COM 4
z= ximc.get_device_name(dev_enum, 3) ## COM 5

def test_controller_names():
    '''prints the com devices the library finds and enumerates'''
    print("Enumerated Devices: " + str(x), str(y), str(z))

def encode_controller_name(motor):
    '''encodes the motor name into a form that open_device can read'''
    if type(motor) is str:
        motor = motor.encode()
    return motor

motor_x= ximc.open_device(encode_controller_name(x))
motor_y= ximc.open_device(encode_controller_name(y))
motor_z= ximc.open_device(encode_controller_name(z))