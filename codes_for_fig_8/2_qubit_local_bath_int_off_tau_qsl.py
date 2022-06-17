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

dim = 4

# parameters
w1 = 3.0
w2 = 3.1
wa = 2.0
wb = 2.1

# delta = 4

e1 = 2.4
e2 = 2.5

# Bath parameters:
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



# Quantum speed limit calculations

tau = np.linspace(0.01, 2, 100)
ti_dim = 10
qsl = np.zeros((len(tau)))

for i in range(len(tau)):
    qsl_theta = np.arccos(np.trace(rho0@rho(tau[i]))/np.trace(rho0@rho0))
    numer = (2*(qsl_theta**2)/math.pi**2)*np.sqrt(np.trace(rho0@rho0))
    numer = numer.real
    
    # Calculation for the denominator 
    ti_dim = ti_dim + 2
    ti = np.linspace(0, tau[i], ti_dim)

    denom_t = np.zeros((ti_dim))
    for j in range(ti_dim):
        E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16 = kraus(ti[j])
        if j<ti_dim-1:
            E1_f, E2_f, E3_f, E4_f, E5_f, E6_f, E7_f, E8_f, E9_f, E10_f, E11_f, E12_f, E13_f, E14_f, E15_f, E16_f = kraus(ti[j+1])
            E1_dt = (E1_f - E1)/(ti[j+1]- ti[j])
            E2_dt = (E2_f - E2)/(ti[j+1]- ti[j])
            E3_dt = (E3_f - E3)/(ti[j+1]- ti[j])
            E4_dt = (E4_f - E4)/(ti[j+1]- ti[j])
            E5_dt = (E5_f - E5)/(ti[j+1]- ti[j])
            E6_dt = (E6_f - E6)/(ti[j+1]- ti[j])
            E7_dt = (E7_f - E7)/(ti[j+1]- ti[j])
            E8_dt = (E8_f - E8)/(ti[j+1]- ti[j])
            E9_dt = (E9_f - E9)/(ti[j+1]- ti[j])
            E10_dt = (E10_f - E10)/(ti[j+1]- ti[j])
            E11_dt = (E11_f - E11)/(ti[j+1]- ti[j])
            E12_dt = (E12_f - E12)/(ti[j+1]- ti[j])
            E13_dt = (E13_f - E13)/(ti[j+1]- ti[j])
            E14_dt = (E14_f - E14)/(ti[j+1]- ti[j])
            E15_dt = (E15_f - E15)/(ti[j+1]- ti[j])
            E16_dt = (E16_f - E16)/(ti[j+1]- ti[j])
        if j==ti_dim-1:
            E1_b, E2_b, E3_b, E4_b, E5_b, E6_b, E7_b, E8_b, E9_b, E10_b, E11_b, E12_b, E13_b, E14_b, E15_b, E16_b = kraus(ti[j-1])
            E1_dt = (E1 - E1_b)/(ti[j]-ti[j-1])
            E2_dt = (E2 - E2_b)/(ti[j]-ti[j-1])
            E3_dt = (E3 - E3_b)/(ti[j]-ti[j-1])
            E4_dt = (E4 - E4_b)/(ti[j]-ti[j-1])
            E5_dt = (E5 - E5_b)/(ti[j]-ti[j-1])
            E6_dt = (E6 - E6_b)/(ti[j]-ti[j-1])
            E7_dt = (E7 - E7_b)/(ti[j]-ti[j-1])
            E8_dt = (E8 - E8_b)/(ti[j]-ti[j-1])
            E9_dt = (E9 - E9_b)/(ti[j]-ti[j-1])
            E10_dt = (E10 - E10_b)/(ti[j]-ti[j-1])
            E11_dt = (E11 - E11_b)/(ti[j]-ti[j-1])
            E12_dt = (E12 - E12_b)/(ti[j]-ti[j-1])
            E13_dt = (E13 - E13_b)/(ti[j]-ti[j-1])
            E14_dt = (E14 - E14_b)/(ti[j]-ti[j-1])
            E15_dt = (E15 - E15_b)/(ti[j]-ti[j-1])
            E16_dt = (E16 - E16_b)/(ti[j]-ti[j-1])
            

        E1_dt_dag = np.conjugate(E1_dt.T)
        E2_dt_dag = np.conjugate(E2_dt.T)
        E3_dt_dag = np.conjugate(E3_dt.T)
        E4_dt_dag = np.conjugate(E4_dt.T)
        E5_dt_dag = np.conjugate(E5_dt.T)
        E6_dt_dag = np.conjugate(E6_dt.T)
        E7_dt_dag = np.conjugate(E7_dt.T)
        E8_dt_dag = np.conjugate(E8_dt.T)
        E9_dt_dag = np.conjugate(E9_dt.T)
        E10_dt_dag = np.conjugate(E10_dt.T)
        E11_dt_dag = np.conjugate(E11_dt.T)
        E12_dt_dag = np.conjugate(E12_dt.T)
        E13_dt_dag = np.conjugate(E13_dt.T)
        E14_dt_dag = np.conjugate(E14_dt.T)
        E15_dt_dag = np.conjugate(E15_dt.T)
        E16_dt_dag = np.conjugate(E16_dt.T)

        denom1 = E1@rho0@E1_dt_dag
        denom2 = E2@rho0@E2_dt_dag
        denom3 = E3@rho0@E3_dt_dag
        denom4 = E4@rho0@E4_dt_dag
        denom5 = E4@rho0@E5_dt_dag
        denom6 = E4@rho0@E6_dt_dag
        denom7 = E4@rho0@E7_dt_dag
        denom8 = E4@rho0@E8_dt_dag
        denom9 = E4@rho0@E9_dt_dag
        denom10 = E4@rho0@E10_dt_dag
        denom11 = E4@rho0@E11_dt_dag
        denom12 = E4@rho0@E12_dt_dag
        denom13 = E4@rho0@E13_dt_dag
        denom14 = E4@rho0@E14_dt_dag
        denom15 = E4@rho0@E15_dt_dag
        denom16 = E4@rho0@E16_dt_dag

        denom_t[j] = np.sqrt(np.trace(np.conjugate(denom1.T)@denom1)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom2.T)@denom2)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom3.T)@denom3)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom4.T)@denom4)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom5.T)@denom5)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom6.T)@denom6)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom7.T)@denom7)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom8.T)@denom8)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom9.T)@denom9)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom10.T)@denom10)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom11.T)@denom11)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom12.T)@denom12)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom13.T)@denom13)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom14.T)@denom14)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom15.T)@denom15)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom16.T)@denom16)).real

    denom_integ = simpson(denom_t, ti)
    denom_integ = (1/tau[i])*denom_integ

    qsl[i] = numer/(denom_integ.real)



data = np.zeros((len(tau), 2))
data[:, 0] = tau
data[:, 1] = qsl
np.savetxt("qsl_local_int_off_tau", data)
