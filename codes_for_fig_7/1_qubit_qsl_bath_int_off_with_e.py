# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 02:57:20 2022

@author: Devvrat Tiwari
"""
import numpy as np 
import math
import cmath
import matplotlib.pyplot as plt
from scipy.integrate import simpson

dim = 2

# parameters
N = 500
w0 = 2
w = 2
T = 1 

# iota 
iota = complex(0, 1)


# initial state 

rho0 = np.zeros((dim, dim))
rho0[0, 0] = 1


# partition function
Z = 0 

for n in range(N+1):
    factor = -(w/T)*(n/(2*N) - 0.5)
    Z = Z + math.exp(factor)
    # end of loop 

print(Z)

# function for alpha(t)
def alpha(t, e):
    alpha_t = 0 
    for n in range(N+1):
        factor = -(w/T)*(n/(2*N) - 0.5)
        eta = math.sqrt((w0 - w/(2*N))**2 + 4*(e**2)*(n+1)*(1 - n/(2*N)))
        alpha_t = alpha_t + 4*(n+1)*(e**2)*(1 - n/(2*N))*(math.sin(eta*t/2)/eta)**2*math.exp(factor)
        # end of loop 
    alpha_t = (1/Z)*alpha_t
    return alpha_t

# function for beta(t)

def beta(t, e):
    beta_t = 0 
    for n in range(N+1):
        factor = -(w/T)*(n/(2*N) - 0.5)
        etap = math.sqrt((w0 - w/(2*N))**2 + 4*(e**2)*n*(1 - (n-1)/(2*N)))
        beta_t = beta_t + 4*n*(e**2)*(1 - (n-1)/(2*N))*(math.sin(etap*t/2)/etap)**2*math.exp(factor)
        # end of loop  
    beta_t = (1/Z)*beta_t
    return beta_t

# definition for Delta(t)

def Delta(t, e):
    delta_t = 0 
    for n in range(N+1):
        factor = -(w/T)*(n/(2*N) - 0.5)
        eta = math.sqrt((w0 - w/(2*N))**2 + 4*(e**2)*(n+1)*(1 - n/(2*N)))
        etap = math.sqrt((w0 - w/(2*N))**2 + 4*(e**2)*n*(1 - (n-1)/(2*N)))
        term1 = math.cos(eta*t/2) - iota*((w0 - w)/(2*N))*math.sin(eta*t/2)
        term2 = math.cos(etap*t/2) + iota*((w0 - w)/(2*N))*math.sin(etap*t/2)
        delta_t = delta_t + cmath.exp(-iota*w*t/(2*N))*term1*term2*factor 
        # end of loop 
    delta_t = (1/Z)*delta_t 
    return delta_t 

# function for rho(t)
def rho(t, e):
    # Initialization of rho(t)
    rhot = np.zeros((dim, dim), dtype = complex)
    
    rhot[0, 0] = rho0[0, 0]*(1 - alpha(t, e)) + rho0[1, 1]*beta(t, e)
    rhot[1, 1] = 1 - rhot[0, 0]
    rhot[0, 1] = rho0[0, 1]*Delta(t, e)
    rhot[1, 0] = np.conjugate(rhot[0, 1])
    return rhot

# Function for the Kraus operators
def kraus(t, e):
    K1 = np.zeros((dim, dim), dtype=complex)
    K2 = np.zeros((dim, dim), dtype=complex)
    K3 = np.zeros((dim, dim), dtype=complex)
    K4 = np.zeros((dim, dim), dtype=complex)
    
    X1 = (1 - (alpha(t, e)+beta(t, e))/2) + (1/2)*math.sqrt((alpha(t, e) - beta(t, e))**2 + 4*abs(Delta(t, e))**2)
    X2 = (1 - (alpha(t, e)+beta(t, e))/2) - (1/2)*math.sqrt((alpha(t, e) - beta(t, e))**2 + 4*abs(Delta(t, e))**2)
    Y1 = (math.sqrt((alpha(t, e) - beta(t, e))**2 + 4*abs(Delta(t, e))**2) - (alpha(t, e) - beta(t, e)))/(2*abs(Delta(t, e)))
    Y2 = (math.sqrt((alpha(t, e) - beta(t, e))**2 + 4*abs(Delta(t, e))**2) + (alpha(t, e) - beta(t, e)))/(2*abs(Delta(t, e)))
    
    theta = np.arctan(Delta(t, e).imag/Delta(t, e).real)
    
    K1[0, 1] = math.sqrt(beta(t, e))
    K2[1, 0] = math.sqrt(alpha(t, e))
    
    k3_fact = math.sqrt(X1/(1 + Y1**2))
    k4_fact = math.sqrt(X2/(1 + Y2**2))
    
    K3[0, 0] = k3_fact*Y1*cmath.exp(iota*theta)
    K3[1, 1] = k3_fact     
    
    K4[0, 0] = k4_fact*Y2*cmath.exp(iota*theta)
    K4[1, 1] = k4_fact
    
    return K1, K2, K3, K4


# Check for Kraus Ops

# for i in range(len(ti)): 
#     K1, K2, K3, K4 = kraus(ti[i])
#     rho_temp1 = K1@rho0@np.conjugate(K1.T)
#     rho_temp1 = rho_temp1 + K2@rho0@np.conjugate(K2.T)
#     rho_temp1 = rho_temp1 + K3@rho0@np.conjugate(K3.T)
#     rho_temp1 = rho_temp1 + K4@rho0@np.conjugate(K4.T)
    
#     rho_temp2 = rho(ti[i])
    
#     print(np.allclose(rho_temp1, rho_temp2))
#     # End of the loop
    

# Quantum speed limit calculations

# Vary with paramater epsilon 

e = np.linspace(0.01, 5, 100)

tau = 1
ti_dim = 100
qsl = np.zeros((len(e)))

for i in range(len(e)):
    qsl_theta = np.arccos(np.trace(rho0@rho(tau, e[i]))/np.trace(rho0@rho0))
    numer = (2*(qsl_theta**2)/math.pi**2)*np.sqrt(np.trace(rho0@rho0))
    numer = numer.real
    
    # Calculation for the denominator 
    ti = np.linspace(0, tau, ti_dim)

    denom_t = np.zeros((ti_dim))
    for j in range(ti_dim):
        K1, K2, K3, K4 = kraus(ti[j], e[i])
        if j<ti_dim-1:
            K1_f, K2_f, K3_f, K4_f = kraus(ti[j+1], e[i])
            K1_dt = (K1_f - K1)/(ti[j+1]- ti[j])
            K2_dt = (K2_f - K2)/(ti[j+1]- ti[j])
            K3_dt = (K3_f - K3)/(ti[j+1]- ti[j])
            K4_dt = (K4_f - K4)/(ti[j+1]- ti[j])
        if j==ti_dim-1:
            K1_b, K2_b, K3_b, K4_b = kraus(ti[j-1], e[i])
            K1_dt = (K1 - K1_b)/(ti[j]-ti[j-1])
            K2_dt = (K2 - K2_b)/(ti[j]-ti[j-1])
            K3_dt = (K3 - K3_b)/(ti[j]-ti[j-1])
            K4_dt = (K4 - K4_b)/(ti[j]-ti[j-1])

        K1_dt_dag = np.conjugate(K1_dt.T)
        K2_dt_dag = np.conjugate(K2_dt.T)
        K3_dt_dag = np.conjugate(K3_dt.T)
        K4_dt_dag = np.conjugate(K4_dt.T)

        denom1 = K1@rho0@K1_dt_dag
        denom2 = K2@rho0@K2_dt_dag
        denom3 = K3@rho0@K3_dt_dag
        denom4 = K4@rho0@K4_dt_dag

        denom_t[j] = np.sqrt(np.trace(np.conjugate(denom1.T)@denom1)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom2.T)@denom2)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom3.T)@denom3)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom4.T)@denom4)).real

    denom_integ = simpson(denom_t, ti)
    denom_integ = (1/tau)*denom_integ

    qsl[i] = numer/(denom_integ.real)
    
    print(i)


data = np.zeros((len(e), 2))
data[:, 0] = e
data[:, 1] = qsl

np.savetxt(f"data_qsl_{N}_e_int_off", data)

    
    
