# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:16:33 2022

@author: Devvrat Tiwari
"""

import numpy as np
import math
from odeintw import odeintw
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import cmath
from scipy.integrate import simpson

# First we define all the parameters below:
w1 = 2.0
w2 = 1.9
wa = 1.1
wb = 1.2


e1 = 2.6
e2 = 2.5

# Bath parameters:
M = 15
N = 15
T = 1

# First we complete the partition function:
factn = 0
factm = 0
Z = 0

for m in range(M+1):
    for n in range(N+1):
        factm = 0.5*(m/M -1)
        factn = 0.5*(n/N -1)
        Z = Z + math.exp((-wa / T) * factm) * math.exp((-wb / T) * factn)

print(Z)

# Next we are going to initialize the density matrix at time t = 0 and for dim = 4:

dim = 4


psi0 = np.zeros((dim,1))

psi0[0, 0] = 1/math.sqrt(2)
psi0[1, 0] = 0
psi0[2, 0] = 0
psi0[3, 0] = 1/math.sqrt(2)

print(psi0)
rho0 = psi0@np.conjugate(psi0.T)

print(rho0)

# First we will find the solution of the differential equations for A1, B1 and C1.
iota = complex(0, 1)

def odesabc(abc, t, m, n, delta):
    df_dt = np.zeros((3), dtype = complex)
    facta = 0
    factb = 0
    factc = 0
    facta = (w1+w2+delta)/2 + wa*(1 - m/(2*M)) + wb*(1-n/(2*N))
    factb = (e2/2)*math.sqrt(1 - n/(2*N))*(n+1)
    factc = (e1/2)*math.sqrt(1 - m/(2*M))*(m+1)
    dA1_dt = -iota*facta*abc[0] - iota*factb*abc[1] - iota*factc*abc[2]

    facta = 0
    factb = 0
    factc = 0

    factb = (w1-w2-delta)/2 + wa*(1 -m/(2*M)) + wb*(1 - (n+1)/2*N)
    facta = (e2/2)*math.sqrt(1 - n/(2*N))
    dB1_dt = -iota*factb*abc[1] - iota*facta*abc[0]

    facta = 0

    factc = (-w1+w2-delta)/2 + wa*(1 - (m+1)/(2*M)) + wb*(1 - n/(2*N))
    facta = (e1/2)*math.sqrt(1 - m/(2*M))
    dC1_dt = -iota*factc*abc[2] - iota*facta*abc[0]
    
    df_dt[0] = dA1_dt
    df_dt[1] = dB1_dt
    df_dt[2] = dC1_dt

    return df_dt


#Below we have defined the function to solve odes D1 E1 and F1

def odesedf(edf, t, m, n, delta):
    df_dt = np.zeros((3), dtype = complex)
    factd = 0
    facte = 0
    factf = 0
    factd = (-w1-w2+delta)/2 + wa*(1 - m/(2*M)) + wb*(1-n/(2*N))
    facte = (e2/2)*math.sqrt(1 - (n-1)/(2*N))*n
    factf = (e1/2)*math.sqrt(1 - (m-1)/(2*M))*m
    dD1_dt = -iota*factd*edf[0] - iota*facte*edf[1] - iota*factf*edf[2]

    factd = 0
    facte = 0
    factf = 0

    facte = (-w1+w2-delta)/2 + wa*(1 - m/(2*M)) + wb*(1 - (n-1)/(2*N))
    factd = (e2/2)*math.sqrt(1 - (n-1)/(2*N))
    dE1_dt = -iota*facte*edf[1] - iota*factd*edf[0]

    factd = 0

    factf = (w1-w2-delta)/2 + wa*(1 - (m-1)/(2*M)) + wb*(1 - n/(2*N))
    factd = (e1/2)*math.sqrt(1 - (m-1)/(2*M))
    dF1_dt = -iota*factf*edf[2] - iota*factd*edf[0]
    
    df_dt[0] = dD1_dt
    df_dt[1] = dE1_dt
    df_dt[2] = dF1_dt
    
    return df_dt


# Below we are going to solve the ODEs for GHI functions 

def odesghi(ghi, t, m, n, delta):
    df_dt = np.zeros((3), dtype = complex)
    factg = 0
    facth = 0
    facti = 0
    factg = (-w1+w2-delta)/2 + wa*(1 - m/(2*M)) + wb*(1 - n/(2*N))
    facth = (e2/2)*math.sqrt(1 - n/(2*N))*(n+1)
    facti = (e1/2)*math.sqrt(1 - (m-1)/(2*M))*m
    dG1_dt = -iota*factg*ghi[0] - iota*facth*ghi[1] - iota*facti*ghi[2]

    factg = 0
    facth = 0
    facti = 0

    facth = (-w1-w2+delta)/2 + wa*(1 - m/(2*M)) + wb*(1- (n+1)/(2*N))
    factg = (e2/2)*math.sqrt(1 - n/(2*N))
    dH1_dt = -iota*facth*ghi[1] - iota*factg*ghi[0]

    factg = 0

    facti = (w1+w2+delta)/2 + wa*(1 - (m-1)/(2*M)) + wb*(1-n/(2*N))
    factg = (e1/2)*math.sqrt(1 - (m-1)/(2*M))
    dI1_dt = -iota*facti*ghi[2] - iota*factg*ghi[0]
    
    df_dt[0] = dG1_dt
    df_dt[1] = dH1_dt
    df_dt[2] = dI1_dt
    
    return df_dt



# Below we are going to solve the ODEs for the JKL functions

def odesjkl(jkl, t, m, n, delta):
    df_dt = np.zeros((3), dtype = complex)
    factj = 0
    factk = 0
    factl = 0
    factj = (w1-w2-delta)/2 + wa*(1 - m/(2*N)) + wb*(1 - n/(2*N))
    factk = (e2/2)*math.sqrt(1 - (n-1)/(2*N))*n
    factl = (e1/2)*math.sqrt(1 - m/(2*M))*(m+1)
    dJ1_dt = -iota*factj*jkl[0] - iota*factk*jkl[1] - iota*factl*jkl[2]

    factj = 0
    factk = 0
    factl = 0

    factk = (w1+w2+delta)/2 + wa*(1 - m/(2*M)) + wb*(1 - (n-1)/(2*N))
    factj = (e2/2)*math.sqrt(1 - (n-1)/(2*N))
    dK1_dt = -iota*factk*jkl[1] - iota*factj*jkl[0]

    factj = 0

    factl = (-w1-w2+delta)/2 + wa*(1 - (m+1)/(2*M)) + wb*(1-n/(2*N))
    factj = (e1/2)*math.sqrt(1 - m/(2*M))
    dL1_dt = -iota*factl*jkl[2] - iota*factj*jkl[0]
    
    df_dt[0] = dJ1_dt
    df_dt[1] = dK1_dt
    df_dt[2] = dL1_dt
    
    return df_dt



# Quantum speed limit Calculation

tau = 6

delta = np.linspace(0, 10, 100)

t_dim = 100
qsl= np.zeros(len(delta))

for ii in range(len(delta)):

    t = np.linspace(0, tau, t_dim)

    denom_t = np.zeros((t_dim))
    rho_tau = np.zeros((t_dim))

    # Below we have used the solution of A1, B1 and C1 to form the first term in rho11

    modA1 = np.zeros((len(t)))
    modB1 = np.zeros((len(t)))
    modC1 = np.zeros((len(t)))
    modD1 = np.zeros((len(t)))
    modE1 = np.zeros((len(t)))
    modF1 = np.zeros((len(t)))
    modG1 = np.zeros((len(t)))
    modH1 = np.zeros((len(t)))
    modI1 = np.zeros((len(t)))
    modJ1 = np.zeros((len(t)))
    modK1 = np.zeros((len(t)))
    modL1 = np.zeros((len(t)))

    A1J1 = np.zeros((len(t)), dtype=complex)
    A1G1 = np.zeros((len(t)), dtype=complex)
    A1D1 = np.zeros((len(t)), dtype=complex)
    J1G1 = np.zeros((len(t)), dtype=complex)
    J1D1 = np.zeros((len(t)), dtype=complex)
    G1D1 = np.zeros((len(t)), dtype=complex)




    for m in range(M+1):
        for n in range(N+1):
            abc0 = [1+0j, 0+0j, 0+0j]
            edf0 = [1+0j, 0+0j, 0+0j]
            ghi0 = [1+0j, 0+0j, 0+0j]
            jkl0 = [1+0j, 0+0j, 0+0j]
            resabc = odeintw(odesabc, abc0, t, args=(m, n, delta[ii],))
            resedf = odeintw(odesedf, edf0, t, args=(m, n, delta[ii],))
            resghi = odeintw(odesghi, ghi0, t, args=(m, n, delta[ii],))
            resjkl = odeintw(odesjkl, jkl0, t, args=(m, n, delta[ii],))
            factm = 0.5*(m/M -1)
            factn = 0.5*(n/N -1)
            
            modA1_temp = np.square(np.abs(resabc[:, 0]))
            modA1_temp = modA1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modA1 = modA1 + modA1_temp
            
            modI1_temp = np.square(np.abs(resghi[:,2]))
            modI1_temp = m*modI1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modI1 = modI1 + modI1_temp
            
            modK1_temp = np.square(np.abs(resjkl[:,1]))
            modK1_temp = n*modK1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modK1 = modK1 + modK1_temp
            
            modB1_temp = np.square(np.abs(resabc[:,1]))
            modB1_temp = (n+1)*modB1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modB1 = modB1 + modB1_temp
            
            modC1_temp = np.square(np.abs(resabc[:,2]))
            modC1_temp = (m+1)*modC1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modC1 = modC1 + modC1_temp
            
            modJ1_temp = np.square(np.abs(resjkl[:,0]))
            modJ1_temp = modJ1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modJ1 = modJ1 + modJ1_temp
            
            modF1_temp = np.square(np.abs(resedf[:,2]))
            modF1_temp = m*modF1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modF1 = modF1 + modF1_temp
            
            modG1_temp = np.square(np.abs(resghi[:,0]))
            modG1_temp = modG1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modG1 = modG1 + modG1_temp
            
            modE1_temp = np.square(np.abs(resedf[:,1]))
            modE1_temp = n*modE1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modE1 = modE1 + modE1_temp
            
            modD1_temp = np.square(np.abs(resedf[:,0]))
            modD1_temp = modD1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modD1 = modD1 + modD1_temp
            
            modL1_temp = np.square(np.abs(resjkl[:,2]))
            modL1_temp = (m+1)*modL1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modL1 = modL1 + modL1_temp
            
            modH1_temp = np.square(np.abs(resghi[:,1]))
            modH1_temp = (n+1)*modH1_temp*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            modH1 = modH1 + modH1_temp
            
            for i in range(len(t)):
                A1J1[i] = A1J1[i] + (resabc[i,0]*np.conjugate(resjkl[i,0]))*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            
                A1G1[i] = A1G1[i] + (resabc[i,0]*np.conjugate(resghi[i,0]))*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
            
                A1D1[i] = A1D1[i] + (resabc[i,0]*np.conjugate(resedf[i,0]))*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
                
                J1G1[i] = J1G1[i] + (resjkl[i,0]*np.conjugate(resghi[i,0]))*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
                
                J1D1[i] = J1D1[i] + (resjkl[i,0]*np.conjugate(resedf[i,0]))*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)
                
                G1D1[i] = G1D1[i] + (resghi[i,0]*np.conjugate(resedf[i,0]))*math.exp((-wa/T)*factm)*math.exp((-wb/T)*factn)


    A1sJ1 = np.conjugate(A1J1)
    A1sG1 = np.conjugate(A1G1)
    A1sD1 = np.conjugate(A1D1)
    J1sG1 = np.conjugate(J1G1)
    J1sD1 = np.conjugate(J1D1)
    G1sD1 = np.conjugate(G1D1)


    # Below we will form the Choi-Jamiolkowski matrix for the operator-sum representation:

    def kraus(i):
        cj = np.zeros((dim*dim, dim*dim), dtype=complex)
        cj[0, 0] = modA1[i]
        cj[0, 5] = A1J1[i]
        cj[0, 10] = A1G1[i]
        cj[0, 15] = A1D1[i]
        cj[1, 1] = modB1[i]
        cj[2, 2] = modC1[i]
        cj[4, 4] = modK1[i]
        cj[5, 0] = A1sJ1[i]
        cj[5, 5] = modJ1[i]
        cj[5, 10] = J1G1[i]
        cj[5, 15] = J1D1[i]
        cj[7, 7] = modL1[i]
        cj[8, 8] = modI1[i]
        cj[10, 0] = A1sG1[i]
        cj[10, 5] = J1sG1[i]
        cj[10, 10] = modG1[i]
        cj[10, 15] = G1D1[i]
        cj[11, 11] = modH1[i]
        cj[13, 13] = modF1[i]
        cj[14, 14] = modE1[i]
        cj[15, 0] = A1sD1[i]
        cj[15, 5] = J1sD1[i]
        cj[15, 10] = G1sD1[i]
        cj[15, 15] = modD1[i]
        
        cj = (1/Z)*cj
        
        eig, vec = eigh(cj)
        # print(vec[:,0])
        
        K1 = np.zeros((dim, dim), dtype=complex)
        K2 = np.zeros((dim, dim), dtype=complex)
        K3 = np.zeros((dim, dim), dtype=complex)
        K4 = np.zeros((dim, dim), dtype=complex)
        K5 = np.zeros((dim, dim), dtype=complex)
        K6 = np.zeros((dim, dim), dtype=complex)
        K7 = np.zeros((dim, dim), dtype=complex)
        K8 = np.zeros((dim, dim), dtype=complex)
        K9 = np.zeros((dim, dim), dtype=complex)
        K10 = np.zeros((dim, dim), dtype=complex)
        K11 = np.zeros((dim, dim), dtype=complex)
        K12 = np.zeros((dim, dim), dtype=complex)
        K13 = np.zeros((dim, dim), dtype=complex)
        K14 = np.zeros((dim, dim), dtype=complex)
        K15 = np.zeros((dim, dim), dtype=complex)
        K16 = np.zeros((dim, dim), dtype=complex)
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K1[j, u] = vec[k, 0]
                k = k+1
    
        K1 = cmath.sqrt(eig[0])*K1

        k = 0 
        for u in range(dim):
            for j in range(dim):
                K2[j, u] = vec[k, 1]
                k = k+1
        
        K2 = cmath.sqrt(eig[1])*K2
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K3[j, u] = vec[k, 2]
                k = k+1
        
        K3 = cmath.sqrt(eig[2])*K3
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K4[j, u] = vec[k, 3]
                k = k+1
        
        K4 = cmath.sqrt(eig[3])*K4
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K5[j, u] = vec[k, 4]
                k = k+1
        
        K5 = cmath.sqrt(eig[4])*K5
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K6[j, u] = vec[k, 5]
                k = k+1
        
        K6 = cmath.sqrt(eig[5])*K6
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K7[j, u] = vec[k, 6]
                k = k+1
        
        K7 = cmath.sqrt(eig[6])*K7
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K8[j, u] = vec[k, 7]
                k = k+1
        
        K8 = cmath.sqrt(eig[7])*K8
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K9[j, u] = vec[k, 8]
                k = k+1
        
        K9 = cmath.sqrt(eig[8])*K9
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K10[j, u] = vec[k, 9]
                k = k+1
        
        K10 = cmath.sqrt(eig[9])*K10
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K11[j, u] = vec[k, 10]
                k = k+1
        
        K11 = cmath.sqrt(eig[10])*K11
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K12[j, u] = vec[k, 11]
                k = k+1
        
        K12 = cmath.sqrt(eig[11])*K12
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K13[j, u] = vec[k, 12]
                k = k+1
        
        K13 = cmath.sqrt(eig[12])*K13
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K14[j, u] = vec[k, 13]
                k = k+1
        
        K14 = cmath.sqrt(eig[13])*K14
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K15[j, u] = vec[k, 14]
                k = k+1
        
        K15 = cmath.sqrt(eig[14])*K15
        
        k = 0 
        for u in range(dim):
            for j in range(dim):
                K16[j, u] = vec[k, 15]
                k = k+1
        
        K16 = cmath.sqrt(eig[15])*K16
        
        return K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K15, K16
        
    rho_tau = np.zeros((dim, dim), dtype=complex)
    denom_t = np.zeros((t_dim))
    # Now we will write the operator-sum representations using Kraus operators from K1 to K16
    for i in range(t_dim):
        K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K15, K16 = kraus(i)
        if i==t_dim-1:
            K1_b, K2_b, K3_b, K4_b, K5_b, K6_b, K7_b, K8_b, K9_b, K10_b, K11_b, K12_b, K13_b, K14_b, K15_b, K16_b = kraus(i-1)
            rho_tau = K1@rho0@np.conjugate(np.transpose(K1))
            rho_tau = rho_tau + K2@rho0@np.conjugate(np.transpose(K2))
            rho_tau = rho_tau + K3@rho0@np.conjugate(np.transpose(K3))
            rho_tau = rho_tau + K4@rho0@np.conjugate(np.transpose(K4))
            rho_tau = rho_tau + K5@rho0@np.conjugate(np.transpose(K5))
            rho_tau = rho_tau + K6@rho0@np.conjugate(np.transpose(K6))
            rho_tau = rho_tau + K7@rho0@np.conjugate(np.transpose(K7))
            rho_tau = rho_tau + K8@rho0@np.conjugate(np.transpose(K8))
            rho_tau = rho_tau + K9@rho0@np.conjugate(np.transpose(K9))
            rho_tau = rho_tau + K10@rho0@np.conjugate(np.transpose(K10))
            rho_tau = rho_tau + K11@rho0@np.conjugate(np.transpose(K11))
            rho_tau = rho_tau + K12@rho0@np.conjugate(np.transpose(K12))
            rho_tau = rho_tau + K13@rho0@np.conjugate(np.transpose(K13))
            rho_tau = rho_tau + K14@rho0@np.conjugate(np.transpose(K14))
            rho_tau = rho_tau + K15@rho0@np.conjugate(np.transpose(K15))
            rho_tau = rho_tau + K16@rho0@np.conjugate(np.transpose(K16))
            
            K1_dt = (K1 - K1_b)/(t[i]- t[i-1])
            K2_dt = (K2 - K2_b)/(t[i]- t[i-1])
            K3_dt = (K3 - K3_b)/(t[i]- t[i-1])
            K4_dt = (K4 - K4_b)/(t[i]- t[i-1])
            K5_dt = (K5 - K5_b)/(t[i]- t[i-1])
            K6_dt = (K6 - K6_b)/(t[i]- t[i-1])
            K7_dt = (K7 - K7_b)/(t[i]- t[i-1])
            K8_dt = (K8 - K8_b)/(t[i]- t[i-1])
            K9_dt = (K9 - K9_b)/(t[i]- t[i-1])
            K10_dt = (K10 - K10_b)/(t[i]- t[i-1])
            K11_dt = (K11 - K11_b)/(t[i]- t[i-1])
            K12_dt = (K12 - K12_b)/(t[i]- t[i-1])
            K13_dt = (K13 - K13_b)/(t[i]- t[i-1])
            K14_dt = (K14 - K14_b)/(t[i]- t[i-1])
            K15_dt = (K15 - K15_b)/(t[i]- t[i-1])
            K16_dt = (K16 - K16_b)/(t[i]- t[i-1])

        if i<t_dim-1:
            K1_f, K2_f, K3_f, K4_f, K5_f, K6_f, K7_f, K8_f, K9_f, K10_f, K11_f, K12_f, K13_f, K14_f, K15_f, K16_f = kraus(i+1)
            K1_dt = (K1_f - K1)/(t[i+1]-t[i])
            K2_dt = (K2_f - K2)/(t[i+1]-t[i])
            K3_dt = (K3_f - K3)/(t[i+1]-t[i])
            K4_dt = (K4_f - K4)/(t[i+1]-t[i])
            K5_dt = (K5_f - K5)/(t[i+1]-t[i])
            K6_dt = (K6_f - K6)/(t[i+1]-t[i])
            K7_dt = (K7_f - K7)/(t[i+1]-t[i])
            K8_dt = (K8_f - K8)/(t[i+1]-t[i])
            K9_dt = (K9_f - K9)/(t[i+1]-t[i])
            K10_dt = (K10_f - K10)/(t[i+1]-t[i])
            K11_dt = (K11_f - K11)/(t[i+1]-t[i])
            K12_dt = (K12_f - K12)/(t[i+1]-t[i])
            K13_dt = (K13_f - K13)/(t[i+1]-t[i])
            K14_dt = (K14_f - K14)/(t[i+1]-t[i])
            K15_dt = (K15_f - K15)/(t[i+1]-t[i])
            K16_dt = (K16_f - K16)/(t[i+1]-t[i])

        K1_dt_dag = np.conjugate(K1_dt.T)
        K2_dt_dag = np.conjugate(K2_dt.T)
        K3_dt_dag = np.conjugate(K3_dt.T)
        K4_dt_dag = np.conjugate(K4_dt.T)
        K5_dt_dag = np.conjugate(K5_dt.T)
        K6_dt_dag = np.conjugate(K6_dt.T)
        K7_dt_dag = np.conjugate(K7_dt.T)
        K8_dt_dag = np.conjugate(K8_dt.T)
        K9_dt_dag = np.conjugate(K9_dt.T)
        K10_dt_dag = np.conjugate(K10_dt.T)
        K11_dt_dag = np.conjugate(K11_dt.T)
        K12_dt_dag = np.conjugate(K12_dt.T)
        K13_dt_dag = np.conjugate(K13_dt.T)
        K14_dt_dag = np.conjugate(K14_dt.T)
        K15_dt_dag = np.conjugate(K15_dt.T)
        K16_dt_dag = np.conjugate(K16_dt.T)

        denom1 = K1@rho0@K1_dt_dag
        denom2 = K2@rho0@K2_dt_dag
        denom3 = K3@rho0@K3_dt_dag
        denom4 = K4@rho0@K4_dt_dag
        denom5 = K5@rho0@K5_dt_dag
        denom6 = K6@rho0@K6_dt_dag
        denom7 = K7@rho0@K7_dt_dag
        denom8 = K8@rho0@K8_dt_dag
        denom9 = K9@rho0@K9_dt_dag
        denom10 = K10@rho0@K10_dt_dag
        denom11 = K11@rho0@K11_dt_dag
        denom12 = K12@rho0@K12_dt_dag
        denom13 = K13@rho0@K13_dt_dag
        denom14 = K14@rho0@K14_dt_dag
        denom15 = K15@rho0@K15_dt_dag
        denom16 = K16@rho0@K16_dt_dag

        denom_t[i] = np.sqrt(np.trace(np.conjugate(denom1.T)@denom1)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom2.T)@denom2)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom3.T)@denom3)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom4.T)@denom4)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom5.T)@denom5)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom6.T)@denom6)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom7.T)@denom7)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom8.T)@denom8)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom9.T)@denom9)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom10.T)@denom10)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom11.T)@denom11)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom12.T)@denom12)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom13.T)@denom13)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom14.T)@denom14)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom15.T)@denom15)).real
        denom_t[i] = denom_t[i] + np.sqrt(np.trace(np.conjugate(denom16.T)@denom16)).real
        # End of the loop for i

    
    qsl_theta = np.arccos(np.trace(rho0@rho_tau)/np.trace(rho0@rho0))
    numer = (2*(qsl_theta**2)/math.pi**2)*np.sqrt(np.trace(rho0@rho0))
    numer = numer.real

    denom_integ = simpson(denom_t, t)
    denom_integ = (1/tau)*denom_integ

    qsl[ii] = numer/(denom_integ.real)
    print(ii)

    # End of the ii loop


data = np.zeros((len(delta), 2))

data[:, 0] = delta
data[:, 1] = qsl

np.savetxt("qsl_global_int_off_e", data)

