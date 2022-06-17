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


t_dim = 100
t = np.linspace(0.01, 2, t_dim)
concur = np.zeros((t_dim))
discord = np.zeros((t_dim))

sy = np.array([[0, -iota], [iota, 0]])
sy_2 = np.kron(sy, sy)

for i in range(t_dim):
    rho_temp = rho(t[i])

    rho_tilde = sy_2@np.conjugate(rho_temp)@sy_2
    rho_temp = sqrtm(sqrtm(rho_temp)@rho_tilde@sqrtm(rho_temp))
    
    eigvalues, v = eigh(rho_temp)

    if ((eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])>0):
        concur[i] = (eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])
    else: 
        concur[i] = 0

    # Calculation of quantum discord
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

    print(i)

    # End of the loop i for time t


data = np.zeros((len(t), 3))
data[:, 0] = t
data[:, 1] = concur
data[:, 2] = discord

np.savetxt("concur_discord_local_int_on_tau", data)

