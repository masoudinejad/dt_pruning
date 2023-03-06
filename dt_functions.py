from ExtendDT import ext_dt
import numpy as np

def leaves_count(base_dt):
    X_dt = ext_dt(base_dt)
    X_dt.findLeafs()
    leaves_count = np.sum(X_dt.is_leaf)
    return leaves_count