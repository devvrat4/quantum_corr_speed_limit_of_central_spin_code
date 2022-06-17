# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 02:57:20 2022

@author: Devvrat Tiwari
"""
import numpy as np 
import math
import cmath
import matplotlib.pyplot as plt
# from scipy.linalg import eigvals
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
    factm = -(wa/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
    Z1 = Z1 + math.exp(factm)
    # end of loop
print(Z1)

Z2 = 0 

for n in range(N+1):
    factn = -(wb/T)*(n*(1 - (n-1)/(2*N)) - 1/2)
    Z2 = Z2 + math.exp(factn)
    # end of loop 
print(Z2)


# Next we will define the A1(n, t), B1(n, t), C1(n, t) and D1(n, t) individually for m and n.
def A1m(m, t):
    eta = np.sqrt((w1 - wa*(1 - m/M))**2 + 4*(e1**2)*(m+1)*(1 - m/(2*M)))
    out = cmath.exp(-iota*t*m*wa*(1 - m/(2*M)))*(-iota*(w1 - wa*(1 - m/M))*(np.sin(eta*t/2)/eta)+ np.cos(eta*t/2))
    return out

def A1n(n, t):
    eta = np.sqrt((w2 - wb*(1 - n/N))**2 + 4*(e2**2)*(n+1)*(1 - n/(2*N)))
    out = cmath.exp(-iota*t*n*wb*(1 - n/(2*N)))*(-iota*(w2 - wb*(1 - n/N))*(np.sin(eta*t/2)/eta)+ np.cos(eta*t/2))
    return out

def B1m(m, t):
    eta = np.sqrt((w1 - wa*(1 - m/M))**2 + 4*(e1**2)*(m+1)*(1 - m/(2*M)))
    out = cmath.exp(-iota*t*m*wa*(1 - m/(2*M)))*(-2*iota*e1*np.sqrt(1 - m/(2*M)))*(np.sin(eta*t/2)/eta)
    return out

def B1n(n, t):
    eta = np.sqrt((w2 - wb*(1 - n/N))**2 + 4*(e2**2)*(n+1)*(1 - n/(2*N)))
    out = cmath.exp(-iota*t*n*wb*(1 - n/(2*N)))*(-2*iota*e2*np.sqrt(1 - n/(2*N)))*(np.sin(eta*t/2)/eta)
    return out

def C1m(m, t):
    etap = np.sqrt((w1 - wa*(1 - (m-1)/M))**2 + 4*(e1**2)*m*(1 - (m-1)/(2*M)))
    out = cmath.exp(-iota*t*wa*(m-1)*(1 - m/M))*(iota*(w1 - wa*(1 - (m-1)/M))*(np.sin(etap*t/2)/etap) + np.cos(etap*t/2))
    return out

def C1n(n, t):
    etap = np.sqrt((w2 - wb*(1 - (n-1)/N))**2 + 4*(e2**2)*n*(1 - (n-1)/(2*N)))
    out = cmath.exp(-iota*t*wb*(n-1)*(1 - n/N))*(iota*(w2 - wb*(1 - (n-1)/N))*(np.sin(etap*t/2)/etap) + np.cos(etap*t/2))
    return out

def D1m(m, t):
    etap = np.sqrt((w1 - wa*(1 - (m-1)/M))**2 + 4*(e1**2)*m*(1 - (m-1)/(2*M)))
    out = cmath.exp(-iota*t*wa*(m-1)*(1 - m/M))*(-2*iota*e1*np.sqrt(1 - (m-1)/(2*M)))*(np.sin(etap*t/2)/etap)
    return out

def D1n(n, t):
    etap = np.sqrt((w2 - wb*(1 - (n-1)/N))**2 + 4*(e2**2)*n*(1 - (n-1)/(2*N)))
    out = cmath.exp(-iota*t*wb*(n-1)*(1 - n/N))*(-2*iota*e2*np.sqrt(1 - (n-1)/(2*N)))*(np.sin(etap*t/2)/etap)
    return out

# Next we will be using a1m for bath A and a1n for bath B

def a1m(t):
    modA1m = 0
    for m in range(M+1):
        factm = -(wa/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modA1m = modA1m + np.exp(factm)*(abs(A1m(m, t)))**2
    modA1m = (1/Z1)*modA1m
    return modA1m

def a1n(t):
    modA1n = 0
    for n in range(N+1):
        factn = -(wb/T)*(n*(1 - (n-1)/(2*N)) - 1/2)
        modA1n = modA1n + np.exp(factn)*(abs(A1n(n, t)))**2
    modA1n = (1/Z2)*modA1n
    return modA1n

def b1m(t):
    modB1m = 0
    for m in range(M+1):
        factm = -(wa/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modB1m = modB1m + (m+1)*np.exp(factm)*(abs(B1m(m, t)))**2
    modB1m = (1/Z1)*modB1m
    return modB1m

def b1n(t):
    modB1n = 0
    for n in range(N+1):
        factn = -(wb/T)*(n*(1 - (n-1)/(2*N)) - 1/2)
        modB1n = modB1n + (n+1)*np.exp(factn)*(abs(B1n(n, t)))**2
    modB1n = (1/Z2)*modB1n
    return modB1n

def c1m(t):
    modC1m = 0
    for m in range(M+1):
        factm = -(wa/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modC1m = modC1m + np.exp(factm)*(abs(C1m(m, t)))**2
    modC1m = (1/Z1)*modC1m
    return modC1m

def c1n(t):
    modC1n = 0
    for n in range(N+1):
        factn = -(wb/T)*(n*(1 - (n-1)/(2*N)) - 1/2)
        modC1n = modC1n + np.exp(factn)*(abs(C1n(n, t)))**2
    modC1n = (1/Z2)*modC1n
    return modC1n

def g1m(t):
    A1C1s = 0
    for m in range(M+1):
        factm = -(wa/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        A1C1s = A1C1s + A1m(m, t)*np.conjugate(C1m(m, t))*np.exp(factm)
    A1C1s = (1/Z1)*A1C1s
    return A1C1s

def g1n(t):
    A1C1s = 0
    for n in range(N+1):
        factn = -(wb/T)*(n*(1 - (n-1)/(2*N)) - 1/2)
        A1C1s = A1C1s + A1n(n, t)*np.conjugate(C1n(n, t))*np.exp(factn)
    A1C1s = (1/Z2)*A1C1s
    return A1C1s

def d1m(t):
    modD1m = 0
    for m in range(M + 1):
        factm = -(wa/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modD1m = modD1m + m*np.exp(factm)*(abs(D1m(m, t)))**2
    modD1m = (1/Z1)*modD1m
    return modD1m

def d1n(t):
    modD1n = 0
    for n in range(N+1):
        factn = -(wb/T)*(n*(1 - (n-1)/(2*N)) - 1/2)
        modD1n = modD1n + n*np.exp(factn)*(abs(D1n(n, t)))**2
    modD1n = (1/Z2)*modD1n
    return modD1n


# Function for the Kraus operators
def kraus(t):
    # For system 1 (M)
    K1m = np.zeros((2, 2), dtype=complex)
    K2m = np.zeros((2, 2), dtype=complex)
    K3m = np.zeros((2, 2), dtype=complex)
    K4m = np.zeros((2, 2), dtype=complex)

    K1m[0, 1] = np.sqrt(d1m(t))
    K2m[1, 0] = np.sqrt(b1m(t))

    R1m = (1/2)*(a1m(t) + c1m(t) + np.sqrt((a1m(t) - c1m(t))**2 + 4*(abs(g1m(t)))**2))
    R2m = (1/2)*(a1m(t) + c1m(t) - np.sqrt((a1m(t) - c1m(t))**2 + 4*(abs(g1m(t)))**2))
    T1m = (0.5/abs(g1m(t)))*(a1m(t) - c1m(t) + np.sqrt((a1m(t) - c1m(t))**2 + 4*(abs(g1m(t)))**2))
    T2m = (0.5/abs(g1m(t)))*(a1m(t) - c1m(t) - np.sqrt((a1m(t) - c1m(t))**2 + 4*(abs(g1m(t)))**2))

    thetam = np.arctan((g1m(t).imag)/(g1m(t).real))

    K3m[0, 0] = T1m*cmath.exp(iota*thetam)*np.sqrt(R1m/(1+T1m**2))
    K3m[1, 1] = np.sqrt(R1m/(1+T1m**2))

    K4m[0, 0] = T2m*cmath.exp(iota*thetam)*np.sqrt(R2m/(1+T2m**2))
    K4m[1, 1] = np.sqrt(R2m/(1+T2m**2))
    
    # For System 2 (N)
    K1n = np.zeros((2, 2), dtype=complex)
    K2n = np.zeros((2, 2), dtype=complex)
    K3n = np.zeros((2, 2), dtype=complex)
    K4n = np.zeros((2, 2), dtype=complex)

    K1n[0, 1] = np.sqrt(d1n(t))
    K2n[1, 0] = np.sqrt(b1n(t))

    R1n = (1/2)*(a1n(t) + c1n(t) + np.sqrt((a1n(t) - c1n(t))**2 + 4*(abs(g1n(t)))**2))
    R2n = (1/2)*(a1n(t) + c1n(t) - np.sqrt((a1n(t) - c1n(t))**2 + 4*(abs(g1n(t)))**2))
    T1n = (0.5/abs(g1n(t)))*(a1n(t) - c1n(t) + np.sqrt((a1n(t) - c1n(t))**2 + 4*(abs(g1n(t)))**2))
    T2n = (0.5/abs(g1n(t)))*(a1n(t) - c1n(t) - np.sqrt((a1n(t) - c1n(t))**2 + 4*(abs(g1n(t)))**2))

    thetan = np.arctan((g1n(t).imag)/(g1n(t).real))

    K3n[0, 0] = T1n*cmath.exp(iota*thetan)*np.sqrt(R1n/(1+T1n**2))
    K3n[1, 1] = np.sqrt(R1n/(1+T1n**2))

    K4n[0, 0] = T2n*cmath.exp(iota*thetan)*np.sqrt(R2n/(1+T2n**2))
    K4n[1, 1] = np.sqrt(R2n/(1+T2n**2))


    # Now Kraus operators for Two qubits
    E1 = np.kron(K1m, K1n)
    E2 = np.kron(K1m, K2n)
    E3 = np.kron(K1m, K3n)
    E4 = np.kron(K1m, K4n)
    E5 = np.kron(K2m, K1n)
    E6 = np.kron(K2m, K2n)
    E7 = np.kron(K2m, K3n)
    E8 = np.kron(K2m, K4n)
    E9 = np.kron(K3m, K1n)
    E10 = np.kron(K3m, K2n)
    E11 = np.kron(K3m, K3n)
    E12 = np.kron(K3m, K4n)
    E13 = np.kron(K4m, K1n)
    E14 = np.kron(K4m, K2n)
    E15 = np.kron(K4m, K3n)
    E16 = np.kron(K4m, K4n)

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
# for i in range(len(ti)):
#     E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16 = kraus(ti[i])
#
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
#
#     if i==60:
#         print(np.allclose(np.eye(4), Eye.real))
#         print(Eye.real)


# Calculation of the quantum speed limit time
tau = np.linspace(0.01, 2, 100)
t_dim = 10
qsl = np.zeros((len(tau)))

for i in range(len(tau)):
    qsl_theta = np.arccos(np.trace(rho0@rho(tau[i]))/np.trace(rho0@rho0))
    numer = ((2*qsl_theta**2)/(math.pi**2)) * np.sqrt(np.trace(rho0 @ rho0))

    # Calculation for the denominator
    t_dim = t_dim + 2
    t = np.linspace(0, tau[i], t_dim)

    denom_t = np.zeros((t_dim))
    for j in range(t_dim):
        E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16 = kraus(t[j])
        if j < t_dim - 1:
            E1_f, E2_f, E3_f, E4_f, E5_f, E6_f, E7_f, E8_f, E9_f, E10_f, E11_f, E12_f, E13_f, E14_f, E15_f, E16_f = kraus(t[j + 1])
            E1_dt = (E1_f - E1) / (t[j + 1] - t[j])
            E2_dt = (E2_f - E2) / (t[j + 1] - t[j])
            E3_dt = (E3_f - E3) / (t[j + 1] - t[j])
            E4_dt = (E4_f - E4) / (t[j + 1] - t[j])
            E5_dt = (E5_f - E5) / (t[j + 1] - t[j])
            E6_dt = (E6_f - E6) / (t[j + 1] - t[j])
            E7_dt = (E7_f - E7) / (t[j + 1] - t[j])
            E8_dt = (E8_f - E8) / (t[j + 1] - t[j])
            E9_dt = (E9_f - E9) / (t[j + 1] - t[j])
            E10_dt = (E10_f - E10) / (t[j + 1] - t[j])
            E11_dt = (E11_f - E11) / (t[j + 1] - t[j])
            E12_dt = (E12_f - E12) / (t[j + 1] - t[j])
            E13_dt = (E13_f - E13) / (t[j + 1] - t[j])
            E14_dt = (E14_f - E14) / (t[j + 1] - t[j])
            E15_dt = (E15_f - E15) / (t[j + 1] - t[j])
            E16_dt = (E16_f - E16) / (t[j + 1] - t[j])
        if j == t_dim - 1:
            E1_b, E2_b, E3_b, E4_b, E5_b, E6_b, E7_b, E8_b, E9_b, E10_b, E11_b, E12_b, E13_b, E14_b, E15_b, E16_b = kraus(t[j - 1])
            E1_dt = (E1 - E1_b) / (t[j] - t[j - 1])
            E2_dt = (E2 - E2_b) / (t[j] - t[j - 1])
            E3_dt = (E3 - E3_b) / (t[j] - t[j - 1])
            E4_dt = (E4 - E4_b) / (t[j] - t[j - 1])
            E5_dt = (E5 - E5_b) / (t[j] - t[j - 1])
            E6_dt = (E6 - E6_b) / (t[j] - t[j - 1])
            E7_dt = (E7 - E7_b) / (t[j] - t[j - 1])
            E8_dt = (E8 - E8_b) / (t[j] - t[j - 1])
            E9_dt = (E9 - E9_b) / (t[j] - t[j - 1])
            E10_dt = (E10 - E10_b) / (t[j] - t[j - 1])
            E11_dt = (E11 - E11_b) / (t[j] - t[j - 1])
            E12_dt = (E12 - E12_b) / (t[j] - t[j - 1])
            E13_dt = (E13 - E13_b) / (t[j] - t[j - 1])
            E14_dt = (E14 - E14_b) / (t[j] - t[j - 1])
            E15_dt = (E15 - E15_b) / (t[j] - t[j - 1])
            E16_dt = (E16 - E16_b) / (t[j] - t[j - 1])

        denom1 = E1 @ rho0 @ np.conjugate(E1_dt.T)
        denom2 = E2 @ rho0 @ np.conjugate(E2_dt.T)
        denom3 = E3 @ rho0 @ np.conjugate(E3_dt.T)
        denom4 = E4 @ rho0 @ np.conjugate(E4_dt.T)
        denom5 = E5 @ rho0 @ np.conjugate(E5_dt.T)
        denom6 = E6 @ rho0 @ np.conjugate(E6_dt.T)
        denom7 = E7 @ rho0 @ np.conjugate(E7_dt.T)
        denom8 = E8 @ rho0 @ np.conjugate(E8_dt.T)
        denom9 = E9 @ rho0 @ np.conjugate(E9_dt.T)
        denom10 = E10 @ rho0 @ np.conjugate(E10_dt.T)
        denom11 = E11 @ rho0 @ np.conjugate(E11_dt.T)
        denom12 = E12 @ rho0 @ np.conjugate(E12_dt.T)
        denom13 = E13 @ rho0 @ np.conjugate(E13_dt.T)
        denom14 = E14 @ rho0 @ np.conjugate(E14_dt.T)
        denom15 = E15 @ rho0 @ np.conjugate(E15_dt.T)
        denom16 = E16 @ rho0 @ np.conjugate(E16_dt.T)

        denom_t[j] = np.sqrt(np.trace(np.conjugate(denom1.T) @ denom1)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom2.T) @ denom2)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom3.T) @ denom3)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom4.T) @ denom4)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom5.T) @ denom5)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom6.T) @ denom6)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom7.T) @ denom7)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom8.T) @ denom8)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom9.T) @ denom9)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom10.T) @ denom10)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom11.T) @ denom11)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom12.T) @ denom12)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom13.T) @ denom13)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom14.T) @ denom14)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom15.T) @ denom15)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom16.T) @ denom16)).real

    denom_integ = simpson(denom_t, t)
    denom_integ = (1/tau[i])*denom_integ

    print(i)

    qsl[i] = numer.real/denom_integ.real

    # End of the loop i for tau

data = np.zeros((len(tau), 2))
data[:, 0] = tau
data[:, 1] = qsl
np.savetxt("qsl_local_int_on_tau", data)


