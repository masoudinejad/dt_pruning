from ExtendDT import ext_dt
import numpy as np

def get_APD(base_dt):
    xdt = ext_dt(base_dt)
    if not hasattr(xdt, "DT_APD"):
        xdt.getRCFactor()
    return xdt.DT_APD

def get_branchSamples(base_dt, node_id):
    xdt = ext_dt(base_dt)
    if not hasattr(xdt, "RC_node"):
        xdt.getRCFactor()
    return xdt.RC_node[node_id]

def get_branchAPD(base_dt, node_id):
    xdt = ext_dt(base_dt)
    if not hasattr(xdt, "RC_branch"):
        xdt.getRCFactor()
    return xdt.RC_branch[node_id]