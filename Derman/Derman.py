import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def derman(surface,s):
    K = np.array(surface.index,dtype=float)
    T = np.array(surface.columns,dtype=int)
    atm_vols = pd.Series(surface.loc[s,:].values,index=T)
    derman_coefs = {}
    for t in T.tolist():
        try:
            vols = surface.loc[:,t].dropna()
            k = vols.index
            x = np.array(vols.index)-s
            y = vols - atm_vols.loc[t]
            model = LinearRegression(fit_intercept=False)
            x = x.reshape(-1,1)
            model.fit(x,y)
            b = model.coef_[0]
            derman_coefs[int(t)] = float(b)

            # estimated = pd.Series(np.asarray(k-s)*b + atm_vols.loc[t],index=k)
            # plt.figure()
            # plt.plot(estimated,color='purple', label='estimated')
            # plt.scatter(vols.index,vols,color = 'green',label='actual')
            # plt.title(f'{t} day maturity')
            # plt.ylabel('implied volatility')
            # plt.xlabel('strike price')
            # plt.show()
            
        except Exception as e:
            print(f"error for {t}: {e}")
            pass

    return derman_coefs

