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
from scipy.linalg import sqrtm


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

Z_on = 0 
for n in range(N+1):
    fact = -(w/T)*(n*(1- (n-1)/(2*N))-1/2)
    Z_on = Z_on + math.exp(fact)

print(Z_on)

rho0 = np.zeros((dim, dim))
rho0[0, 0] = 1
rho0[0, 1] = 0
rho0[1, 0] = 0
rho0[1, 1] = 0
print(rho0)

iota = complex(0, 1)

# Physical quantities to check

tr = np.zeros((len(t)))
rho_11 = np.zeros((len(t)))
rho_12 = np.zeros((len(t)), dtype=complex)
rho_21 = np.zeros((len(t)), dtype=complex)
rho_22 = np.zeros((len(t)))
tr_dist_on_off = np.zeros((len(t)))

def A1(n, t):
    exp_fact = -iota*(n*w*(1 - n/(2*N)))
    eta = math.sqrt((w0 - w*(1 - n/N))**2 + 4*(e**2)*(n+1)*(1 - n/N))
    term = -iota*(w0 - w*(1 - n/N))*np.sin(eta*t/2)/eta + np.cos(eta*t/2)
    out = cmath.exp(exp_fact*t)*(term)
    return out

def B1(n, t):
    exp_fact = -iota*(n*w*(1 - n/(2*N)))
    eta = math.sqrt((w0 - w*(1 - n/N))**2 + 4*(e**2)*(n+1)*(1 - n/N))
    term = -2*iota*e*math.sqrt(1 - n/(2*N))*np.sin(eta*t/2)/eta
    out = cmath.exp(exp_fact*t)*term
    return out

def C1(n, t):
    exp_fact = -iota*(w*(n-1)*(1 - n/N))
    etap = math.sqrt((w0 - w*(1 - (n-1)/N))**2 + 4*n*(e**2)*(1 - (n-1)/(2*N)))
    term = iota*(w0 - w*(1 - (n-1)/N))*np.sin(etap*t/2)/etap + np.cos(etap*t/2)
    out = cmath.exp(exp_fact*t)*term
    return out

def D1(n, t):
    exp_fact = -iota*(w*(n-1)*(1 - n/N))
    etap = math.sqrt((w0 - w*(1 - (n-1)/N))**2 + 4*n*(e**2)*(1 - (n-1)/(2*N)))
    term = -2*iota*e*math.sqrt(1 - (n-1)/(2*N))*np.sin(etap*t/2)/etap
    out = cmath.exp(exp_fact*t)*term
    return out



for i in range(len(t)):
    alpha_t = 0
    beta_t = 0 
    Delta_t = 0 
    eta = 0
    eta_p = 0 
    
    # The model with bath interaction off 
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
    
    # rho_11[i] = rho[0, 0].real
    
    # rho_12[i] = rho[0, 1]
    
    # rho_21[i] = rho[1, 0]
    
    # rho_22[i] = rho[1, 1].real
    
    # The model with bath interaction on
    rho_on = np.zeros((dim, dim), dtype=complex)
    
    termA1 = 0
    termB1 = 0
    termC1 = 0
    termD1 = 0
    termA1C1s = 0 
    for n in range(N+1):
        factZ = -(w/T)*(n*(1- (n-1)/(2*N))-1/2)
        termA1 = termA1 + (abs(A1(n, t[i]))**2)*np.exp(factZ)
        termD1 = termD1 + n*(abs(D1(n, t[i]))**2)*np.exp(factZ)
        termC1 = termC1 + (abs(C1(n, t[i]))**2)*np.exp(factZ)
        termB1 = termB1 + (n+1)*(abs(B1(n, t[i]))**2)*np.exp(factZ)
        termA1C1s = termA1C1s + A1(n, t[i])*np.conjugate(C1(n, t[i]))*np.exp(factZ)
    
    rho_on[0, 0] = (1/Z_on)*termA1*rho0[0, 0] + (1/Z_on)*termD1*rho0[1, 1]
    rho_on[0, 1] = (1/Z_on)*termA1C1s*rho0[0, 1]
    rho_on[1, 0] = np.conjugate(rho[0, 1])
    rho_on[1, 1] = 1 - rho[0, 0]

    rho_diff = rho_on - rho

    tr_dist_on_off[i] = 0.5*np.trace(sqrtm((np.conjugate(rho_diff.T)@rho_diff).real))
    
    # End of the loop for i


# data = np.zeros((len(t), 2))
# data[:, 0] = t
# data[:, 1] = tr_dist_on_off
# np.savetxt("trace_dist_single_case", data)

plt.plot(t, tr_dist_on_off, label = r"Trace distance with int on or off")
plt.legend(loc= 4)
plt.grid()
plt.show()
    
