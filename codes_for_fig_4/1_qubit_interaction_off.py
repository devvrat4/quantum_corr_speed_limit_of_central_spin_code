# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 04:38:36 2022

@author: Devvrat Tiwari
"""

import numpy as np
import cmath
import math
# from odeintw import odeintw
import matplotlib.pyplot as plt
from scipy.linalg import eigvals


t = np.linspace(0, 10, 500)


# Parameters 
dim = 2

e = 1

w = 2

w0 = 2

N = 100

T = 1

Z = 0 
for n in range(N+1):
    Z = Z + math.exp((-w/T)*(0.5*(n/N - 1)))

print(Z)

rho0 = np.zeros((dim, dim))
rho0[0, 0] = 1
print(rho0)

iota = complex(0, 1)


tr = np.zeros((len(t)))
rho_11 = np.zeros((len(t)))
rho_12 = np.zeros((len(t)), dtype=complex)
rho_21 = np.zeros((len(t)), dtype=complex)
rho_22 = np.zeros((len(t)))
von_ent = np.zeros((len(t)))


for i in range(len(t)):
    alpha_t = 0
    beta_t = 0 
    Delta_t = 0 
    eta = 0
    eta_p = 0 
    
    rho = np.zeros((dim, dim), dtype=complex)
    
    for n in range(N+1):
        factn = math.exp((-w/T)*(0.5*(n/N - 1)))
        eta = math.sqrt((w0 - 0.5*w/N)**2 + 4*(e**2)*(n+1)*(1 - 0.5*n/N))
        eta_p = math.sqrt((w0 - 0.5*w/N)**2 + 4*(e**2)*n*(1 - 0.5*(n-1)/N))
        
        alpha_t = alpha_t + factn*4*(n+1)*(e**2)*(1 - 0.5*n/N)*(math.sin((eta*t[i])/2)/eta)**2
        
        beta_t = beta_t + factn*4*n*(e**2)*(1 - 0.5*(n-1)/N)*(math.sin((eta_p*t[i])/2)/eta_p)**2
        
        delta_fact1 = math.cos(eta*t[i]/2) - iota*(w0 - 0.5*w/N)*math.sin(eta*t[i]/2)
        delta_fact2 = math.cos(eta_p*t[i]/2) + iota*(w0 - 0.5*w/N)*math.sin(eta_p*t[i]/2)
        Delta_t = Delta_t + factn*cmath.exp((-iota*w*t[i])/(2*N))*delta_fact1*delta_fact2
        
    alpha_t = (1/Z)*alpha_t
    beta_t = (1/Z)*beta_t
    Delta_t = (1/Z)*Delta_t    
    # Now, the rho would be 
    
    rho[0, 0] = (1 - alpha_t)*rho0[0, 0] + beta_t *rho0[1, 1]
    rho[0, 1] = Delta_t * rho0[0, 1] 
    rho[1, 0] = np.conjugate(rho[0, 1])
    rho[1, 1] = 1 - rho[0, 0]
    
    rho_11[i] = rho[0, 0].real
    
    rho_12[i] = rho[0, 1]
    
    rho_21[i] = rho[1, 0]
    
    rho_22[i] = rho[1, 1].real
    
    
    tr[i] = np.trace(rho).real
    
    eigenvs = eigvals(rho)
    
    for u in range(len(eigenvs)):
        if(abs(eigenvs[u])>1e-14):
            von_ent[i] = von_ent[i] - eigenvs[u].real*math.log(eigenvs[u].real, 2)
    
data = np.zeros((len(t), 2))
data[:, 0] = t
data[:, 1] = von_ent   
np.savetxt("von_ent_int_off", data)

    
