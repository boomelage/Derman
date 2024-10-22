import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def linearize_vol_surface(surface,s,atm_vols):
    K = np.array(surface.index).tolist()
    T = np.array(atm_vols.index).tolist()
    
    derman_coefs = pd.Series(np.empty(len(T),dtype=float),index=T)
    for t in T:
        vols = surface.loc[:,t].dropna()
        x = np.array(vols.index)-s
        y = vols - atm_vols.loc[t]
        model = LinearRegression(fit_intercept=False)
        x = x.reshape(-1,1)
        model.fit(x,y)
        b = model.coef_[0]
        derman_coefs.loc[t] = b
    
    derman_surface = pd.DataFrame(
        np.empty((len(K),len(T)),dtype=float),
        index=K,
        columns=T
    )
    
    for k in K:
        moneyness = k-s
        for t in T:
            derman_surface.loc[k,t] = atm_vols.loc[t] + moneyness*derman_coefs.loc[t]
    
    vol_matrix = derman_surface.copy()
    return vol_matrix

