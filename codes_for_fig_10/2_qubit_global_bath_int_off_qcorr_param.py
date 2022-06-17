# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:16:33 2022

@author: Devvrat Tiwari
"""

import numpy as np
import math
from odeintw import odeintw
import matplotlib.pyplot as plt
from scipy.linalg import eigh, sqrtm
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





tau = 6

delta = np.linspace(0, 10, 100)

t_dim = 100
concur = np.zeros(len(delta))
discord = np.zeros(len(delta))

sy = np.array([[0, -iota], [iota, 0]])

sy_2 = np.kron(sy, sy)

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
    K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K15, K16 = kraus(t_dim-1)
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
            
    # Calculation of Concurrence
    rho_tilde = sy_2@np.conjugate(rho_tau)@sy_2
    rho_temp = sqrtm(sqrtm(rho_tau)@rho_tilde@sqrtm(rho_tau))
    
    eigvalues, v = eigh(rho_temp)
    
    # print(eigvalues)
    
    # print(cmath.sqrt(eigvalues[3]) - cmath.sqrt(eigvalues[2]) - cmath.sqrt(eigvalues[1]) - cmath.sqrt(eigvalues[0]))
    
    if ((eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])>0):
        concur[ii] = (eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])
    else: 
        concur[ii] = 0
        
    # Calculation of Quantum Discord
    l1 = (rho_tau[0, 0] + rho_tau[3, 3])/2 + 0.5*cmath.sqrt((rho_tau[0, 0] - rho_tau[3, 3])**2 + 4*abs(rho_tau[0, 3])**2).real
    l2 = (rho_tau[0, 0] + rho_tau[3, 3])/2 - 0.5*cmath.sqrt((rho_tau[0, 0] - rho_tau[3, 3])**2 + 4*abs(rho_tau[0, 3])**2).real
    l5 = 1 + np.sqrt((rho_tau[0, 0]+rho_tau[1, 1]-rho_tau[2, 2]-rho_tau[3, 3])**2 + 4*abs(rho_tau[0, 3])**2)
    l6 = 1 - np.sqrt((rho_tau[0, 0]+rho_tau[1, 1]-rho_tau[2, 2]-rho_tau[3, 3])**2 + 4*abs(rho_tau[0, 3])**2)
    if (rho_tau[0, 0]>0 and rho_tau[3, 3]>0 and l1>0 and l2>0):
        discord_term1 = -(rho_tau[0, 0]+rho_tau[2, 2])*math.log(rho_tau[0, 0]+rho_tau[2, 2])/math.log(2) - (rho_tau[1, 1]+rho_tau[3, 3])*math.log(rho_tau[1, 1]+rho_tau[3, 3])/math.log(2)
        discord_term2 = rho_tau[1, 1]*math.log(rho_tau[1, 1])/math.log(2) + rho_tau[2, 2]*math.log(rho_tau[2, 2])/math.log(2) + l1*cmath.log(l1)/math.log(2) + l2*math.log(l2)/math.log(2)
        discord_term3 = -(1/2)*l5*math.log(l5/2)/math.log(2) - (1/2)*l6*math.log(l6/2)/math.log(2)
        discord[ii] = discord_term1 + discord_term2 + discord_term3
    
    print(ii)

    # End of the ii loop
    

data = np.zeros((len(delta), 3))

data[:, 0] = delta
data[:, 1] = concur
data[:, 2] = discord

np.savetxt("concur_discord_global_int_off_e", data)


