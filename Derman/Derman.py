import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def linearize_vol_surface(surface,s,atm_vols):
    K = np.array(surface.index)
    T = np.array(atm_vols.index)
    derman_coefs = pd.Series(np.empty(len(T),dtype=float),index=T)
    for t in T:
        vols = surface.loc[:,t].dropna()
        k = vols.index
        x = np.array(vols.index)-s
        y = vols - atm_vols.loc[t]
        model = LinearRegression(fit_intercept=False)
        x = x.reshape(-1,1)
        model.fit(x,y)
        b = model.coef_[0]
        derman_coefs.loc[t] = b
        estimated = pd.Series(np.asarray(k-s)*b + atm_vols.loc[t],index=k)

        # plt.figure()
        # plt.plot(estimated,color='purple', label='estimated')
        # plt.scatter(vols.index,vols,color = 'green',label='actual')
        # plt.title(f'{t} day maturity')
        # plt.ylabel('implied volatility')
        # plt.xlabel('strike price')
        # plt.show()

    return derman_coefs

