# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 02:57:20 2022

@author: Devvrat Tiwari
"""
import numpy as np 
import math
import cmath
import matplotlib.pyplot as plt
from scipy.linalg import eigh, sqrtm

dim = 4

# parameters
w1 = 3.0
w2 = 3.1
wa = 2.0
wb = 2.1


e1 = 2.4
e2 = 2.5

M = 25
N = 25
T = 1

# iota 
iota = complex(0, 1)


# initial state 

rho0 = np.zeros((dim, dim))
rho0[0, 0] = 1/2
rho0[0, 3] = 1/2
rho0[3, 0] = 1/2
rho0[3, 3] = 1/2
print(rho0)

# partition function
Z1 = 0 

for m in range(M+1):
    factor = -(wa/T)*(m/(2*M) - 0.5)
    Z1 = Z1 + math.exp(factor)
    # end of loop

Z2 = 0 

for n in range(N+1):
    factor = -(wb/T)*(n/(2*N) - 0.5)
    Z2 = Z2 + math.exp(factor)
    # end of loop 


# function for alpha(t)
def alpha1(t):
    alpha_t = 0 
    for m in range(M+1):
        factor = -(wa/T)*(m/(2*M) - 0.5)
        eta = math.sqrt((w1 - wa/(2*M))**2 + 4*(e1**2)*(m+1)*(1 - m/(2*M)))
        alpha_t = alpha_t + 4*(m+1)*(e1**2)*(1 - m/(2*M))*(math.sin(eta*t/2)/eta)**2*math.exp(factor)
        # end of loop 
    alpha_t = (1/Z1)*alpha_t
    return alpha_t

def alpha2(t):
    alpha_t = 0 
    for n in range(N+1):
        factor = -(wb/T)*(n/(2*N) - 0.5)
        eta = math.sqrt((w2 - wb/(2*N))**2 + 4*(e2**2)*(n+1)*(1 - n/(2*N)))
        alpha_t = alpha_t + 4*(n+1)*(e2**2)*(1 - n/(2*N))*(math.sin(eta*t/2)/eta)**2*math.exp(factor)
        # end of loop 
    alpha_t = (1/Z2)*alpha_t
    return alpha_t

# function for beta(t)

def beta1(t):
    beta_t = 0 
    for m in range(M+1):
        factor = -(wa/T)*(m/(2*M) - 0.5)
        etap = math.sqrt((w1 - wa/(2*M))**2 + 4*(e1**2)*m*(1 - (m-1)/(2*M)))
        beta_t = beta_t + 4*m*(e1**2)*(1 - (m-1)/(2*M))*(math.sin(etap*t/2)/etap)**2*math.exp(factor)
        # end of loop  
    beta_t = (1/Z1)*beta_t
    return beta_t

def beta2(t):
    beta_t = 0 
    for n in range(N+1):
        factor = -(wb/T)*(n/(2*N) - 0.5)
        etap = math.sqrt((w2 - wb/(2*N))**2 + 4*(e2**2)*n*(1 - (n-1)/(2*N)))
        beta_t = beta_t + 4*n*(e2**2)*(1 - (n-1)/(2*N))*(math.sin(etap*t/2)/etap)**2*math.exp(factor)
        # end of loop  
    beta_t = (1/Z2)*beta_t
    return beta_t

# definition for Delta(t)
def Delta1(t):
    delta_t = 0 
    for m in range(M+1):
        factor = -(wa/T)*(m/(2*M) - 0.5)
        eta = math.sqrt((w1 - wa/(2*M))**2 + 4*(e1**2)*(m+1)*(1 - m/(2*M)))
        etap = math.sqrt((w1 - wa/(2*M))**2 + 4*(e1**2)*m*(1 - (m-1)/(2*M)))
        term1 = math.cos(eta*t/2) - iota*((w1 - wa)/(2*M))*math.sin(eta*t/2)
        term2 = math.cos(etap*t/2) + iota*((w1 - wa)/(2*M))*math.sin(etap*t/2)
        delta_t = delta_t + cmath.exp(-iota*wa*t/(2*N))*term1*term2*factor 
        # end of loop 
    delta_t = (1/Z1)*delta_t 
    return delta_t 

def Delta2(t):
    delta_t = 0 
    for n in range(N+1):
        factor = -(wb/T)*(n/(2*N) - 0.5)
        eta = math.sqrt((w2 - wb/(2*N))**2 + 4*(e2**2)*(n+1)*(1 - n/(2*N)))
        etap = math.sqrt((w2 - wb/(2*N))**2 + 4*(e2**2)*n*(1 - (n-1)/(2*N)))
        term1 = math.cos(eta*t/2) - iota*((w2 - wb)/(2*N))*math.sin(eta*t/2)
        term2 = math.cos(etap*t/2) + iota*((w2 - wb)/(2*N))*math.sin(etap*t/2)
        delta_t = delta_t + cmath.exp(-iota*wb*t/(2*N))*term1*term2*factor 
        # end of loop 
    delta_t = (1/Z2)*delta_t 
    return delta_t 


# Function for the Kraus operators
def kraus(t):
    # For system 1 (M)
    K11 = np.zeros((2, 2), dtype=complex)
    K21 = np.zeros((2, 2), dtype=complex)
    K31 = np.zeros((2, 2), dtype=complex)
    K41 = np.zeros((2, 2), dtype=complex)
    
    X11 = (1 - (alpha1(t)+beta1(t))/2) + (1/2)*math.sqrt((alpha1(t) - beta1(t))**2 + 4*abs(Delta1(t))**2)
    X21 = (1 - (alpha1(t)+beta1(t))/2) - (1/2)*math.sqrt((alpha1(t) - beta1(t))**2 + 4*abs(Delta1(t))**2)
    Y11 = (math.sqrt((alpha1(t) - beta1(t))**2 + 4*abs(Delta1(t))**2) - (alpha1(t) - beta1(t)))/(2*abs(Delta1(t)))
    Y21 = (math.sqrt((alpha1(t) - beta1(t))**2 + 4*abs(Delta1(t))**2) + (alpha1(t) - beta1(t)))/(2*abs(Delta1(t)))
    
    theta1 = np.arctan(Delta1(t).imag/Delta1(t).real)
    
    K11[0, 1] = math.sqrt(beta1(t))
    K21[1, 0] = math.sqrt(alpha1(t))
    
    k31_fact = math.sqrt(X11/(1 + Y11**2))
    k41_fact = math.sqrt(X21/(1 + Y21**2))
    
    K31[0, 0] = k31_fact*Y11*cmath.exp(iota*theta1)
    K31[1, 1] = k31_fact     
    
    K41[0, 0] = k41_fact*Y21*cmath.exp(iota*theta1)
    K41[1, 1] = k41_fact
    
    # For System 2 (N)
    K12 = np.zeros((2, 2), dtype=complex)
    K22 = np.zeros((2, 2), dtype=complex)
    K32 = np.zeros((2, 2), dtype=complex)
    K42 = np.zeros((2, 2), dtype=complex)
    
    X12 = (1 - (alpha2(t)+beta2(t))/2) + (1/2)*math.sqrt((alpha2(t) - beta2(t))**2 + 4*abs(Delta2(t))**2)
    X22 = (1 - (alpha2(t)+beta2(t))/2) - (1/2)*math.sqrt((alpha2(t) - beta2(t))**2 + 4*abs(Delta2(t))**2)
    Y12 = (math.sqrt((alpha2(t) - beta2(t))**2 + 4*abs(Delta2(t))**2) - (alpha2(t) - beta2(t)))/(2*abs(Delta2(t)))
    Y22 = (math.sqrt((alpha2(t) - beta2(t))**2 + 4*abs(Delta2(t))**2) + (alpha2(t) - beta2(t)))/(2*abs(Delta2(t)))
    
    theta2 = np.arctan(Delta2(t).imag/Delta2(t).real)
    
    K12[0, 1] = math.sqrt(beta2(t))
    K22[1, 0] = math.sqrt(alpha2(t))
    
    k32_fact = math.sqrt(X12/(1 + Y12**2))
    k42_fact = math.sqrt(X22/(1 + Y22**2))
    
    K32[0, 0] = k32_fact*Y12*cmath.exp(iota*theta2)
    K32[1, 1] = k32_fact     
    
    K42[0, 0] = k42_fact*Y22*cmath.exp(iota*theta2)
    K42[1, 1] = k42_fact

    # Now Kraus operators for Two qubits
    E1 = np.kron(K11, K12)
    E2 = np.kron(K11, K22)
    E3 = np.kron(K11, K32)
    E4 = np.kron(K11, K42)
    E5 = np.kron(K21, K12)
    E6 = np.kron(K21, K22)
    E7 = np.kron(K21, K32)
    E8 = np.kron(K21, K42)
    E9 = np.kron(K31, K12)
    E10 = np.kron(K31, K22)
    E11 = np.kron(K31, K32)
    E12 = np.kron(K31, K42)
    E13 = np.kron(K41, K12)
    E14 = np.kron(K41, K22)
    E15 = np.kron(K41, K32)
    E16 = np.kron(K41, K42)

    return E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16

# Function for rho(t)

def rho(t):
    E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16 = kraus(t)
    
    rho_t = np.zeros((dim, dim), dtype=complex)

    rho_t = E1@rho0@np.conjugate(E1.T)
    rho_t = rho_t + E2@rho0@np.conjugate(E2.T)
    rho_t = rho_t + E3@rho0@np.conjugate(E3.T)
    rho_t = rho_t + E4@rho0@np.conjugate(E4.T)
    rho_t = rho_t + E5@rho0@np.conjugate(E5.T)
    rho_t = rho_t + E6@rho0@np.conjugate(E6.T)
    rho_t = rho_t + E7@rho0@np.conjugate(E7.T)
    rho_t = rho_t + E8@rho0@np.conjugate(E8.T)
    rho_t = rho_t + E9@rho0@np.conjugate(E9.T)
    rho_t = rho_t + E10@rho0@np.conjugate(E10.T)
    rho_t = rho_t + E11@rho0@np.conjugate(E11.T)
    rho_t = rho_t + E12@rho0@np.conjugate(E12.T)
    rho_t = rho_t + E13@rho0@np.conjugate(E13.T)
    rho_t = rho_t + E14@rho0@np.conjugate(E14.T)
    rho_t = rho_t + E15@rho0@np.conjugate(E15.T)
    rho_t = rho_t + E16@rho0@np.conjugate(E16.T)

    return rho_t


# Check for Kraus Ops
# ti_dim = 100
# ti = np.linspace(0, 10, ti_dim)
# trace = np.zeros((ti_dim))
# rho_11 = np.zeros((ti_dim))
# trace2 = np.zeros((ti_dim))
# for i in range(len(ti)): 
#     E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16 = kraus(ti[i])

#     Eye = np.zeros((dim, dim), dtype=complex)
#     Eye = np.conjugate(E1.T)@E1
#     Eye = Eye + np.conjugate(E2.T)@E2
#     Eye = Eye + np.conjugate(E3.T)@E3
#     Eye = Eye + np.conjugate(E4.T)@E4
#     Eye = Eye + np.conjugate(E5.T)@E5
#     Eye = Eye + np.conjugate(E6.T)@E6
#     Eye = Eye + np.conjugate(E7.T)@E7
#     Eye = Eye + np.conjugate(E8.T)@E8
#     Eye = Eye + np.conjugate(E9.T)@E9
#     Eye = Eye + np.conjugate(E10.T)@E10
#     Eye = Eye + np.conjugate(E11.T)@E11
#     Eye = Eye + np.conjugate(E12.T)@E12
#     Eye = Eye + np.conjugate(E13.T)@E13
#     Eye = Eye + np.conjugate(E14.T)@E14
#     Eye = Eye + np.conjugate(E15.T)@E15
#     Eye = Eye + np.conjugate(E16.T)@E16

#     print(np.allclose(np.eye(4), Eye.real))


# Concurrence and quantum discord calculation 
ti_dim = 100
ti = np.linspace(0.01, 2, ti_dim)
concur = np.zeros((ti_dim))
concur2 = np.zeros((ti_dim))
discord = np.zeros((ti_dim))

sy = np.array([[0, -iota], [iota, 0]])
sy_2 = np.kron(sy, sy)

for i in range(ti_dim):
    rho_temp = rho(ti[i])

    rho_tilde = sy_2@np.conjugate(rho_temp)@sy_2
    rho_temp = sqrtm(sqrtm(rho_temp)@rho_tilde@sqrtm(rho_temp))
    
    eigvalues, v = eigh(rho_temp)
    
    # print(eigvalues)
    
    if ((eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])>0):
        concur[i] = (eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])
    else: 
        concur[i] = 0
    
    term1 = abs(rho_temp[0, 3]) - math.sqrt(rho_temp[1, 1]*rho_temp[2, 2])
    term2 = abs(rho_temp[1, 2]) - math.sqrt(rho_temp[0, 0]*rho_temp[3, 3])
    
    if (term1>term2 and term1>0):
        concur2[i] = 2*term1
    elif (term2>term1 and term2>0):
        concur2[i] = 2*term2
    else:
        concur2[i] = 0
    
    # Calculation of Quantum Dsicord
    l1 = (rho_temp[0, 0] + rho_temp[3, 3])/2 + 0.5*cmath.sqrt((rho_temp[0, 0] - rho_temp[3, 3])**2 + 4*abs(rho_temp[0, 3])**2).real
    l2 = (rho_temp[0, 0] + rho_temp[3, 3])/2 - 0.5*cmath.sqrt((rho_temp[0, 0] - rho_temp[3, 3])**2 + 4*abs(rho_temp[0, 3])**2).real
    l5 = 1 + np.sqrt((rho_temp[0, 0]+rho_temp[1, 1]-rho_temp[2, 2]-rho_temp[3, 3])**2 + 4*abs(rho_temp[0, 3])**2)
    l6 = 1 - np.sqrt((rho_temp[0, 0]+rho_temp[1, 1]-rho_temp[2, 2]-rho_temp[3, 3])**2 + 4*abs(rho_temp[0, 3])**2)
    if (rho_temp[0, 0]>0 and rho_temp[3, 3]>0 and l1>0 and l2>0):
        discord_term1 = -(rho_temp[0, 0]+rho_temp[2, 2])*math.log(rho_temp[0, 0]+rho_temp[2, 2])/math.log(2) - (rho_temp[1, 1]+rho_temp[3, 3])*math.log(rho_temp[1, 1]+rho_temp[3, 3])/math.log(2)
        discord_term2 = rho_temp[1, 1]*math.log(rho_temp[1, 1])/math.log(2) + rho_temp[2, 2]*math.log(rho_temp[2, 2])/math.log(2) + l1*cmath.log(l1)/math.log(2) + l2*math.log(l2)/math.log(2)
        discord_term3 = -(1/2)*l5*math.log(l5/2)/math.log(2) - (1/2)*l6*math.log(l6/2)/math.log(2)
        discord[i] = discord_term1 + discord_term2 + discord_term3
    
    discord[0] = 1


data = np.zeros((len(ti), 3))
data[:, 0] = ti
data[:, 1] = concur
data[:, 2] = discord
np.savetxt("concur_discord_local_int_off_tau", data)
