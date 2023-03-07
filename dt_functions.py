from ExtendDT import ext_dt
import numpy as np

def get_APD(base_dt):
    xdt = ext_dt(base_dt)
    xdt.getRCFactor()
    return xdt.DT_APD