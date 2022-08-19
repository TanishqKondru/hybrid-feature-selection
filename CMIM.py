from entropy_estimators import *
from tqdm import tqdm 
import numpy as np

def cmim(X, y, **kwargs):

    n_samples, n_features = X.shape
    F = []
    J_CMIM = []
    MIfy = []
    
    is_n_selected_features_specified = False

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
    t1 = np.zeros(n_features)
    
    max = -10000000*np.ones(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    j_cmim = 1
    
    p_bar = tqdm(total=n_features)
    while True:
        p_bar.update(1)
        if len(F) == 0:
            idx = np.argmax(t1)
            F.append(idx)
            J_CMIM.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]
            
        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmim <= 0:
                break
        j_cmim = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = midd(f_select, f)
                t3 = cmidd(f_select, f, y)
                if t2-t3 > max[i]:
                        max[i] = t2-t3
                t = t1[i] - max[i]
                if t > j_cmim:
                    j_cmim = t
                    idx = i
                    
        F.append(idx)
        J_CMIM.append(j_cmim)
        MIfy.append(t1[idx])
        f_select = X[:, idx]
        
    p_bar.close()
    return np.array(F), np.array(J_CMIM), np.array(MIfy)