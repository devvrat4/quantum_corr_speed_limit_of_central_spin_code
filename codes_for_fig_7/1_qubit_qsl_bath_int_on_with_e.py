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
from scipy.integrate import simpson


t = np.linspace(0, 10, 500)


# Parameters
dim = 2

# e = 1

w = 2

w0 = 2

M = 500

T = 1

iota = complex(0, 1)

# initial density matrix

rho0 = np.zeros((dim, dim))
rho0[0, 0] = 1

print(rho0)

Z = 0

for m in range(M+1):
    factm = -(w/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
    Z = Z + math.exp(factm)
    # end of loop
print(Z)

# Next we define A1, B1, C1 and D1

def A1(m, t, e):
    eta = np.sqrt((w0 - w*(1 - m/M))**2 + 4*(e**2)*(m+1)*(1 - m/(2*M)))
    out = cmath.exp(-iota*t*m*w*(1 - m/(2*M)))*(-iota*(w0 - w*(1 - m/M))*(np.sin(eta*t/2)/eta)+ np.cos(eta*t/2))
    return out

def B1(m, t, e):
    eta = np.sqrt((w0 - w*(1 - m/M))**2 + 4*(e**2)*(m+1)*(1 - m/(2*M)))
    out = cmath.exp(-iota*t*m*w*(1 - m/(2*M)))*(-2*iota*e*np.sqrt(1 - m/(2*M)))*(np.sin(eta*t/2)/eta)
    return out


def C1(m, t, e):
    etap = np.sqrt((w0 - w*(1 - (m-1)/M))**2 + 4*(e**2)*m*(1 - (m-1)/(2*M)))
    out = cmath.exp(-iota*t*w*(m-1)*(1 - m/M))*(iota*(w0 - w*(1 - (m-1)/M))*(np.sin(etap*t/2)/etap) + np.cos(etap*t/2))
    return out


def D1(m, t, e):
    etap = np.sqrt((w0 - w*(1 - (m-1)/M))**2 + 4*(e**2)*m*(1 - (m-1)/(2*M)))
    out = cmath.exp(-iota*t*w*(m-1)*(1 - m/M))*(-2*iota*e*np.sqrt(1 - (m-1)/(2*M)))*(np.sin(etap*t/2)/etap)
    return out

# Next we define the a1, b1, c1, d1, and gamma1

def a1(t, e):
    modA1m = 0
    for m in range(M+1):
        factm = -(w/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modA1m = modA1m + np.exp(factm)*(abs(A1(m, t, e)))**2
    modA1m = (1/Z)*modA1m
    return modA1m


def b1(t, e):
    modB1m = 0
    for m in range(M+1):
        factm = -(w/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modB1m = modB1m + (m+1)*np.exp(factm)*(abs(B1(m, t, e)))**2
    modB1m = (1/Z)*modB1m
    return modB1m


def c1(t, e):
    modC1m = 0
    for m in range(M+1):
        factm = -(w/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modC1m = modC1m + np.exp(factm)*(abs(C1(m, t, e)))**2
    modC1m = (1/Z)*modC1m
    return modC1m


def g1(t, e):
    A1C1s = 0
    for m in range(M+1):
        factm = -(w/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        A1C1s = A1C1s + A1(m, t, e)*np.conjugate(C1(m, t, e))*np.exp(factm)
    A1C1s = (1/Z)*A1C1s
    return A1C1s


def d1(t, e):
    modD1m = 0
    for m in range(M + 1):
        factm = -(w/T)*(m*(1 - (m-1)/(2*M)) - 1/2)
        modD1m = modD1m + m*np.exp(factm)*(abs(D1(m, t, e)))**2
    modD1m = (1/Z)*modD1m
    return modD1m

# Next we define the Kraus Operators

def kraus(t, e):
    K1 = np.zeros((2, 2), dtype=complex)
    K2 = np.zeros((2, 2), dtype=complex)
    K3 = np.zeros((2, 2), dtype=complex)
    K4 = np.zeros((2, 2), dtype=complex)

    K1[0, 1] = np.sqrt(d1(t, e))
    K2[1, 0] = np.sqrt(b1(t, e))

    R1 = (1/2)*(a1(t, e) + c1(t, e) + np.sqrt((a1(t, e) - c1(t, e))**2 + 4*(abs(g1(t, e)))**2))
    R2 = (1/2)*(a1(t, e) + c1(t, e) - np.sqrt((a1(t, e) - c1(t, e))**2 + 4*(abs(g1(t, e)))**2))
    T1 = (0.5/abs(g1(t, e)))*(a1(t, e) - c1(t, e) + np.sqrt((a1(t, e) - c1(t, e))**2 + 4*(abs(g1(t, e)))**2))
    T2 = (0.5/abs(g1(t, e)))*(a1(t, e) - c1(t, e) - np.sqrt((a1(t, e) - c1(t, e))**2 + 4*(abs(g1(t, e)))**2))

    theta = np.arctan((g1(t, e).imag)/(g1(t, e).real))

    K3[0, 0] = T1*cmath.exp(iota*theta)*np.sqrt(R1/(1+T1**2))
    K3[1, 1] = np.sqrt(R1/(1+T1**2))

    K4[0, 0] = T2*cmath.exp(iota*theta)*np.sqrt(R2/(1+T2**2))
    K4[1, 1] = np.sqrt(R2/(1+T2**2))

    return K1, K2, K3, K4

# To check the Kraus operators

# t = np.linspace(0, 10, 100)

# for i in range(len(t)):
#     K1, K2, K3, K4 = kraus(t[i])
#
#     Eye = np.conjugate(K1.T)@K1
#     Eye = Eye + np.conjugate(K2.T)@K2
#     Eye = Eye + np.conjugate(K3.T) @ K3
#     Eye = Eye + np.conjugate(K4.T) @ K4
#
#     print(np.allclose(np.eye(2), Eye.real))

    # End of the loop (i) for time


def rho(t, e):
    rho_temp = np.zeros((dim, dim), dtype=complex)
    K1, K2, K3, K4 = kraus(t, e)

    rho_temp = K1 @ rho0 @ np.conjugate(K1.T)
    rho_temp = rho_temp + K2 @ rho0 @ np.conjugate(K2.T)
    rho_temp = rho_temp + K3 @ rho0 @ np.conjugate(K3.T)
    rho_temp = rho_temp + K4 @ rho0 @ np.conjugate(K4.T)

    return rho_temp

tau = 1
e = np.linspace(0.01, 5, 100)
t_dim = 100
qsl = np.zeros((len(e)))

for i in range(len(e)):
    qsl_theta = np.arccos(np.trace(rho0 @ rho(tau, e[i]))/np.trace(rho0 @ rho0))
    numer = ((2*qsl_theta**2)/(math.pi**2))*np.sqrt(np.trace(rho0 @ rho0))

    # Calculation for the denominator
    t = np.linspace(0, tau, t_dim)

    denom_t = np.zeros((t_dim))
    for j in range(t_dim):
        K1, K2, K3, K4 = kraus(t[j], e[i])
        if j<t_dim-1:
            K1_f, K2_f, K3_f, K4_f = kraus(t[j+1], e[i])
            K1_dt = (K1_f - K1)/(t[j+1]-t[j])
            K2_dt = (K2_f - K2)/(t[j+1]-t[j])
            K3_dt = (K3_f - K3)/(t[j+1]-t[j])
            K4_dt = (K4_f - K4)/(t[j+1]-t[j])
        if j==t_dim-1:
            K1_b, K2_b, K3_b, K4_b = kraus(t[j-1], e[i])
            K1_dt = (K1 - K1_b)/(t[j] - t[j-1])
            K2_dt = (K2 - K2_b) / (t[j] - t[j - 1])
            K3_dt = (K3 - K3_b) / (t[j] - t[j - 1])
            K4_dt = (K4 - K4_b) / (t[j] - t[j - 1])

        denom1 = K1 @ rho0 @ np.conjugate(K1_dt.T)
        denom2 = K2 @ rho0 @ np.conjugate(K2_dt.T)
        denom3 = K3 @ rho0 @ np.conjugate(K3_dt.T)
        denom4 = K4 @ rho0 @ np.conjugate(K4_dt.T)

        denom_t[j] = np.sqrt(np.trace(np.conjugate(denom1.T) @ denom1)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom2.T) @ denom2)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom3.T) @ denom3)).real
        denom_t[j] = denom_t[j] + np.sqrt(np.trace(np.conjugate(denom4.T) @ denom4)).real

    denom_integ = simpson(denom_t, t)
    denom_integ = (1/tau) * denom_integ

    qsl[i] = numer.real/(denom_integ.real)
    print(i)

    # End of the loop i for tau

data = np.zeros((len(e), 2))
data[:, 0] = e
data[:, 1] = qsl

np.savetxt(f"data_qsl_{M}_e_int_on", data)

