    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm

'''
Asymptotic approximation for 0<beta<=1 by Hagan
'''
def bsm_vol(strike, forward, texp, sigma, alpha=0, rho=0, beta=1):
    if(texp<=0.0):
        return( 0.0 )

    powFwdStrk = (forward*strike)**((1-beta)/2)
    logFwdStrk = np.log(forward/strike)
    logFwdStrk2 = logFwdStrk**2
    rho2 = rho*rho

    pre1 = powFwdStrk*( 1 + (1-beta)**2/24 * logFwdStrk2*(1 + (1-beta)**2/80 * logFwdStrk2) )
  
    pre2alp0 = (2-3*rho2)*alpha**2/24
    pre2alp1 = alpha*rho*beta/4/powFwdStrk
    pre2alp2 = (1-beta)**2/24/powFwdStrk**2

    pre2 = 1 + texp*( pre2alp0 + sigma*(pre2alp1 + pre2alp2*sigma) )

    zz = powFwdStrk*logFwdStrk*alpha/np.fmax(sigma, 1e-32)  # need to make sure sig > 0
    if isinstance(zz, float):
        zz = np.array([zz])
    yy = np.sqrt(1 + zz*(zz-2*rho))

    xx_zz = np.zeros(zz.size)

    ind = np.where(abs(zz) < 1e-5)
    xx_zz[ind] = 1 + (rho/2)*zz[ind] + (1/2*rho2-1/6)*zz[ind]**2 + 1/8*(5*rho2-3)*rho*zz[ind]**3
    ind = np.where(zz >= 1e-5)
    xx_zz[ind] = np.log( (yy[[ind]] + (zz[ind]-rho))/(1-rho) ) / zz[ind]
    ind = np.where(zz <= -1e-5)
    xx_zz[ind] = np.log( (1+rho)/(yy[ind] - (zz[ind]-rho)) ) / zz[ind]

    bsmvol = sigma*pre2/(pre1*xx_zz) # bsm vol
    return(bsmvol[0] if bsmvol.size==1 else bsmvol)

'''
Asymptotic approximation for beta=0 by Hagan
'''
def norm_vol(strike, forward, texp, sigma, alpha=0, rho=0):
    # forward, spot, sigma may be either scalar or np.array. 
    # texp, alpha, rho, beta should be scholar values

    if(texp<=0.0):
        return( 0.0 )
    
    zeta = (forward - strike)*alpha/np.fmax(sigma, 1e-32)
    # explicitly make np.array even if args are all scalar or list
    if isinstance(zeta, float):
        zeta = np.array([zeta])
        
    yy = np.sqrt(1 + zeta*(zeta - 2*rho))
    chi_zeta = np.zeros(zeta.size)
    
    rho2 = rho*rho
    ind = np.where(abs(zeta) < 1e-5)
    chi_zeta[ind] = 1 + 0.5*rho*zeta[ind] + (0.5*rho2 - 1/6)*zeta[ind]**2 + 1/8*(5*rho2-3)*rho*zeta[ind]**3

    ind = np.where(zeta >= 1e-5)
    chi_zeta[ind] = np.log( (yy[ind] + (zeta[ind] - rho))/(1-rho) ) / zeta[ind]

    ind = np.where(zeta <= -1e-5)
    chi_zeta[ind] = np.log( (1+rho)/(yy[ind] - (zeta[ind] - rho)) ) / zeta[ind]

    nvol = sigma * (1 + (2-3*rho2)/24*alpha**2*texp) / chi_zeta
 
    return(nvol[0] if nvol.size==1 else nvol)

'''
Hagan model class for 0<beta<=1
'''
class ModelHagan:
    alpha, beta, rho = 0.0, 1.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.beta = beta
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return bsm_vol(strike, forward, texp, sigma, alpha=self.alpha, beta=self.beta, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        bsm_vol = self.bsm_vol(strike, spot, texp, sigma)
        return self.bsm_model.price(strike, spot, texp, bsm_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        return sigma
    
    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or bsm vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp = self.texp if (texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))

        impliedvol = np.zeros(3)
        if (is_vol):
            impliedvol = price_or_vol3
        else:
            for i in range(3):
                impliedvol[i] = self.bsm_model.impvol(price_or_vol3[i], strike3[i], spot, texp, cp_sign = cp_sign)

        bsmvolfun = lambda _parameter: \
            bsm_vol(strike3, forward, texp, _parameter[0], alpha = _parameter[1], rho = _parameter[2]) - impliedvol
        sol = sopt.root(bsmvolfun, [0.1, 0.1, 0]).x

        return  sol[0], sol[1], sol[2] # sigma, alpha, rho
        

'''
Hagan model class for beta=0
'''
class ModelNormalHagan:
    alpha, beta, rho = 0.0, 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.beta = 0.0 # not used but put it here
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return norm_vol(strike, forward, texp, sigma, alpha=self.alpha, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        n_vol = self.norm_vol(strike, spot, texp, sigma)
        return self.normal_model.price(strike, spot, texp, n_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            norm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 50)
        if(setval):
            self.sigma = sigma
        return sigma

    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or normal vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp = self.texp if (texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))

        impliedvol = np.zeros(3)
        if (is_vol):
            impliedvol = price_or_vol3
        else:
            for i in range(3):
                impliedvol[i] = self.normal_model.impvol(price_or_vol3[i], strike3[i], spot, texp, cp_sign = cp_sign)

        normvolfun = lambda _parameter: \
            norm_vol(strike3, forward, texp, _parameter[0], alpha = _parameter[1], rho = _parameter[2]) - impliedvol
        sol = sopt.root(normvolfun, [0.1*forward, 0.1, 0]).x

        return  sol[0], sol[1], sol[2]  # sigma, alpha, rho

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0, delta_t=0.01, sample = 10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        self.delta_t = delta_t
        self.sample = sample
        
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp, sigma, cp_sign=cp_sign)
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        
        return vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        
        
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if (texp is None) else texp
        disc_fac = np.exp(-texp*self.intr)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        step = int(texp/self.delta_t)
        vol_mc = np.ones([self.sample, step+1])
        
        #simulate volatility
        np.random.seed(123)
        Z_1 = np.random.normal(size=(self.sample, step))
        vol_mc[:,1:] = np.cumsum(self.alpha*np.sqrt(self.delta_t)*Z_1 - 0.5*self.alpha**2*self.delta_t, axis = 1)
        vol_mc = sigma * np.exp(vol_mc[:,:-1])
        
        #simulate price
        np.random.seed(321)
        Z_2 = self.rho * Z_1 + np.sqrt(1 - self.rho**2) * np.random.normal(size = (self.sample, step))
        price_mc = np.cumsum(vol_mc*np.sqrt(self.delta_t)*Z_2 - 0.5*vol_mc**2*self.delta_t, axis = 1)
        price_mc = np.fmax(cp_sign*forward*np.exp(price_mc[:,-1])-strike, 0)
       
        return price_mc.mean()*disc_fac


'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0,delta_t=0.01, sample = 10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        self.delta_t = delta_t
        self.sample = sample
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp, sigma, cp_sign=cp_sign)
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)

        return vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if (texp is None) else texp
        disc_fac = np.exp(-texp*self.intr)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        step = int(texp/self.delta_t)
        vol_mc = np.ones([self.sample, step+1])
        
        #simulate volatility
        np.random.seed(123)
        Z_1 = np.random.normal(size=(self.sample, step))
        vol_mc[:,1:] = np.cumsum(self.alpha*np.sqrt(self.delta_t)*Z_1 - 0.5*self.alpha**2*self.delta_t, axis = 1)
        vol_mc = sigma * np.exp(vol_mc[:,:-1])
        
        #simulate price
        np.random.seed(321)
        Z_2 = self.rho * Z_1 + np.sqrt(1 - self.rho**2) * np.random.normal(size = (self.sample, step))
        price_mc = np.sum(vol_mc*Z_2*np.sqrt(self.delta_t), axis = 1)
        price_mc = np.fmax(cp_sign*forward*price_mc-strike, 0)
       
        return price_mc.mean()*disc_fac

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0, delta_t=0.01, sample = 10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        self.delta_t = delta_t
        self.sample = sample
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        return 0
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if (texp is None) else texp
        step = int(texp/self.delta_t)
        
        #simulate volatility
        np.random.seed(123)
        Z_1 = np.random.normal(size=(self.sample, step))
        vol_mc[:,1:] = np.cumsum(self.alpha*np.sqrt(self.delta_t)*Z_1 - 0.5*self.alpha**2*self.delta_t, axis = 1)
        vol_mc = sigma * np.exp(vol_mc[:,:-1])
        
        #simulate variance  
        sim_w = np.ones((self.sample, self.step + 1))
        sim_w[:,1::2], sim_w[:,2::2], sim_w[:,-1] = 4,2,1
        integ_var = self.delta_t/3 * np.sum(sim_w*vol_mc**2, axis=1)

        forward_mc = spot * np.exp(self.rho/self.alpha*(vol_mc[:,-1]-vol_mc[:,0])-self.rho**2/2*integ_var)
        vol_mc = np.sqrt((1-self.rho**2)*integ_var/texp)

        price = self.bsm_model.price(strike, forward_mc, texp, vol_mc, cp_sign=cp_sign)
        return price.mean()

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0, delta_t=0.01, sample = 10000):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        self.delta_t = delta_t
        self.sample = sample    
        
        
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price = self.price(strike, spot, texp, sigma, cp_sign=cp_sign)
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)

        return vol
    
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if (texp is None) else texp
        step = int(texp/self.delta_t)
        
        #simulate volatility
        np.random.seed(123)
        Z_1 = np.random.normal(size=(self.sample, step))
        vol_mc[:,1:] = np.cumsum(self.alpha*np.sqrt(self.delta_t)*Z_1 - 0.5*self.alpha**2*self.delta_t, axis = 1)
        vol_mc = sigma * np.exp(vol_mc[:,:-1])
        
        #simulate variance  
        sim_w = np.ones((self.sample, self.step + 1))
        sim_w[:,1:-1] = 2
        integ_var = self.delta_t/3 * np.sum(sim_w*vol_mc**2, axis=1)

        forward_mc = self.rho/self.alpha*(vol_mc[:,-1]-vol_mc[:,0])
        vol_mc = np.sqrt((1-self.rho**2)*integ_var/texp)

        price = self.bsm_model.price(strike, forward_mc, texp, vol_mc, cp_sign=cp_sign)
        return price.mean()
    
    