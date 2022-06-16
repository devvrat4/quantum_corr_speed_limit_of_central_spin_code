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
    fact = -(w/T)*(n*(1- (n-1)/(2*N))-1/2)
    Z = Z + math.exp(fact)

print(Z)

rho0 = np.zeros((dim, dim))
rho0[0, 0] = 1
rho0[0, 1] = 0
rho0[1, 0] = 0
rho0[1, 1] = 0

print(rho0)

iota = complex(0, 1)


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


tr = np.zeros((len(t)))
rho_11 = np.zeros((len(t)))
rho_12 = np.zeros((len(t)), dtype=complex)
rho_21 = np.zeros((len(t)), dtype=complex)
rho_22 = np.zeros((len(t)))
von_ent = np.zeros((len(t)))


for i in range(len(t)):
    
    rho = np.zeros((dim, dim), dtype=complex)
    
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

    rho[0, 0] = (1/Z)*termA1*rho0[0, 0] + (1/Z)*termD1*rho0[1, 1]
    rho[0, 1] = (1/Z)*termA1C1s*rho0[0, 1]
    rho[1, 0] = np.conjugate(rho[0, 1])
    rho[1, 1] = 1 - rho[0, 0]# (1/Z)*termB1*rho0[0, 0] + (1/Z)*termC1*rho0[1, 1]
    
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
np.savetxt("von_ent_int_on", data)