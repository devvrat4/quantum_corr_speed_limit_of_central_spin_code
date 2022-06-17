# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:24:12 2022

@author: Devvrat Tiwari
"""

import numpy as np
import math
from odeintw import odeintw
import matplotlib.pyplot as plt
from scipy.linalg import eigh, sqrtm
import cmath

# First we define all the parameters below:
w1 = 2.0
w2 = 1.9
wa = 1.1
wb = 1.2

delta = 2.0

e1 = 2.0
e2 = 2.0

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
        factm = m*(1-(m-1)/(2*M))-1/2
        factn = n*(1-(n-1)/(2*N))-1/2
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
rho0 = psi0@np.transpose(psi0)

print(rho0)

# First we will find the solution of the differential equations for A1, B1 and C1.
iota = complex(0, 1)

def odesabc(abc, t, m, n):
    df_dt = np.zeros((3), dtype = complex)
    facta = 0
    factb = 0
    factc = 0
    facta = (w1+w2+delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    factb = (e2/2)*math.sqrt(1 - n/(2*N))*(n+1)
    factc = (e1/2)*math.sqrt(1 - m/(2*M))*(m+1)
    dA1_dt = -iota*facta*abc[0] - iota*factb*abc[1] - iota*factc*abc[2]

    facta = 0
    factb = 0
    factc = 0

    factb = (w1-w2-delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*((n+1)*(1 - n/(2*N)) - 1/2)
    facta = (e2/2)*math.sqrt(1 - n/(2*N))
    dB1_dt = -iota*factb*abc[1] - iota*facta*abc[0]

    facta = 0

    factc = (-w1+w2-delta)/2 + wa*((m+1)*(1 - m/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    facta = (e1/2)*math.sqrt(1 - m/(2*M))
    dC1_dt = -iota*factc*abc[2] - iota*facta*abc[0]
    
    df_dt[0] = dA1_dt
    df_dt[1] = dB1_dt
    df_dt[2] = dC1_dt

    return df_dt


#Below we have defined the function to solve odes D1 E1 and F1

def odesedf(edf, t, m, n):
    df_dt = np.zeros((3), dtype = complex)
    factd = 0
    facte = 0
    factf = 0
    factd = (-w1-w2+delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    facte = (e2/2)*math.sqrt(1 - (n-1)/(2*N))*n
    factf = (e1/2)*math.sqrt(1 - (m-1)/(2*M))*m
    dD1_dt = -iota*factd*edf[0] - iota*facte*edf[1] - iota*factf*edf[2]

    factd = 0
    facte = 0
    factf = 0

    facte = (-w1+w2-delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*((n-1)*(1 - (n-2)/(2*N)) - 1/2)
    factd = (e2/2)*math.sqrt(1 - (n-1)/(2*N))
    dE1_dt = -iota*facte*edf[1] - iota*factd*edf[0]

    factd = 0

    factf = (w1-w2-delta)/2 + wa*((m-1)*(1 - (m-2)/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    factd = (e1/2)*math.sqrt(1 - (m-1)/(2*M))
    dF1_dt = -iota*factf*edf[2] - iota*factd*edf[0]
    
    df_dt[0] = dD1_dt
    df_dt[1] = dE1_dt
    df_dt[2] = dF1_dt
    
    return df_dt


# Below we are going to solve the ODEs for GHI functions 

def odesghi(ghi, t, m, n):
    df_dt = np.zeros((3), dtype = complex)
    factg = 0
    facth = 0
    facti = 0
    factg = (-w1+w2-delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    facth = (e2/2)*math.sqrt(1 - n/(2*N))*(n+1)
    facti = (e1/2)*math.sqrt(1 - (m-1)/(2*M))*m
    dG1_dt = -iota*factg*ghi[0] - iota*facth*ghi[1] - iota*facti*ghi[2]

    factg = 0
    facth = 0
    facti = 0

    facth = (-w1-w2+delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*((n+1)*(1 - n/(2*N)) - 1/2)
    factg = (e2/2)*math.sqrt(1 - n/(2*N))
    dH1_dt = -iota*facth*ghi[1] - iota*factg*ghi[0]

    factg = 0

    facti = (w1+w2+delta)/2 + wa*((m-1)*(1 - (m-2)/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    factg = (e1/2)*math.sqrt(1 - (m-1)/(2*M))
    dI1_dt = -iota*facti*ghi[2] - iota*factg*ghi[0]
    
    df_dt[0] = dG1_dt
    df_dt[1] = dH1_dt
    df_dt[2] = dI1_dt
    
    return df_dt



# Below we are going to solve the ODEs for the JKL functions

def odesjkl(jkl, t, m, n):
    df_dt = np.zeros((3), dtype = complex)
    factj = 0
    factk = 0
    factl = 0
    factj = (w1-w2-delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    factk = (e2/2)*math.sqrt(1 - (n-1)/(2*N))*n
    factl = (e1/2)*math.sqrt(1 - m/(2*M))*(m+1)
    dJ1_dt = -iota*factj*jkl[0] - iota*factk*jkl[1] - iota*factl*jkl[2]

    factj = 0
    factk = 0
    factl = 0

    factk = (w1+w2+delta)/2 + wa*(m*(1 - (m-1)/(2*M)) - 1/2) + wb*((n-1)*(1 - (n-2)/(2*N)) - 1/2)
    factj = (e2/2)*math.sqrt(1 - (n-1)/(2*N))
    dK1_dt = -iota*factk*jkl[1] - iota*factj*jkl[0]

    factj = 0

    factl = (-w1-w2+delta)/2 + wa*((m+1)*(1 - m/(2*M)) - 1/2) + wb*(n*(1 - (n-1)/(2*N)) - 1/2)
    factj = (e1/2)*math.sqrt(1 - m/(2*M))
    dL1_dt = -iota*factl*jkl[2] - iota*factj*jkl[0]
    
    df_dt[0] = dJ1_dt
    df_dt[1] = dK1_dt
    df_dt[2] = dL1_dt
    
    return df_dt




t = np.linspace(0.01, 5, 50)

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
        resabc = odeintw(odesabc, abc0, t, args=(m, n,))
        resedf = odeintw(odesedf, edf0, t, args=(m, n,))
        resghi = odeintw(odesghi, ghi0, t, args=(m, n,))
        resjkl = odeintw(odesjkl, jkl0, t, args=(m, n,))
        factm = m*(1-(m-1)/(2*M))-(1/2)
        factn = n*(1-(n-1)/(2*N))-(1/2)
        
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
        


rho_11 = (1/Z)*(modA1*rho0[0, 0] + modI1*rho0[2, 2] + modK1*rho0[1, 1])

rho_22 = (1/Z)*(modJ1*rho0[1, 1] + modF1*rho0[3, 3] + modB1*rho0[0, 0]) 

rho_33 = (1/Z)*(modG1*rho0[2, 2] + modE1*rho0[3, 3] + modC1*rho0[0, 0])

rho_44 = (1/Z)*(modD1*rho0[3, 3] + modL1*rho0[1, 1] + modH1*rho0[2, 2])

rho_12 = (1/Z)*(A1J1*rho0[0, 1])

rho_13 = (1/Z)*(A1G1*rho0[0, 2])

rho_14 = (1/Z)*(A1D1*rho0[0, 3])

rho_23 = (1/Z)*(J1G1*rho0[1, 2])

rho_24 = (1/Z)*(J1D1*rho0[1, 3])

rho_34 = (1/Z)*(G1D1*rho0[2, 3])

rho_21 = np.conjugate(rho_12)

rho_31 = np.conjugate(rho_13)

rho_41 = np.conjugate(rho_14)

rho_32 = np.conjugate(rho_23)

rho_42 = np.conjugate(rho_24)

rho_43 = np.conjugate(rho_34)




tr = np.zeros((len(t)))
tr_2 = np.zeros((len(t)))
lin_ent = np.zeros((len(t)))
von_ent = np.zeros((len(t)))
tr_dist = np.zeros((len(t)))
discord = np.zeros((len(t)))



sy = np.array([[0, -iota], [iota, 0]])

sy_2 = np.kron(sy, sy)

print(sy_2)

concur = np.zeros((len(t)))
concur2 = np.zeros((len(t)))


for i in range(len(t)):
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = rho_11[i]
    rho[1, 1] = rho_22[i]
    rho[2, 2] = rho_33[i]
    rho[3, 3] = rho_44[i]
    rho[0, 1] = rho_12[i]
    rho[0, 2] = rho_13[i]
    rho[0, 3] = rho_14[i]
    rho[1, 0] = rho_21[i]
    rho[1, 2] = rho_23[i]
    rho[1, 3] = rho_24[i]
    rho[2, 0] = rho_31[i]
    rho[2, 1] = rho_32[i]
    rho[2, 3] = rho_34[i]
    rho[3, 0] = rho_41[i]
    rho[3, 1] = rho_42[i]
    rho[3, 2] = rho_43[i]
    
    # Calculation of Concurrence
    rho_tilde = sy_2@np.conjugate(rho)@sy_2
    rho_temp = sqrtm(sqrtm(rho)@rho_tilde@sqrtm(rho))
    
    eigvalues, v = eigh(rho_temp)
    
    # print(eigvalues)
    
    # print(cmath.sqrt(eigvalues[3]) - cmath.sqrt(eigvalues[2]) - cmath.sqrt(eigvalues[1]) - cmath.sqrt(eigvalues[0]))
    
    if ((eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])>0):
        concur[i] = (eigvalues[3]) - (eigvalues[2]) -(eigvalues[1]) - (eigvalues[0])
    else: 
        concur[i] = 0
    
    term1 = abs(rho[0, 3]) - math.sqrt(rho[1, 1]*rho[2, 2])
    term2 = abs(rho[1, 2]) - math.sqrt(rho[0, 0]*rho[3, 3])
    
    if (term1>term2 and term1>0):
        concur2[i] = 2*term1
    elif (term2>term1 and term2>0):
        concur2[i] = 2*term2
    else:
        concur2[i] = 0
    
    # Calculation of Quantum Dsicord
    l1 = (rho[0, 0] + rho[3, 3])/2 + 0.5*cmath.sqrt((rho[0, 0] - rho[3, 3])**2 + 4*abs(rho[0, 3])**2).real
    l2 = (rho[0, 0] + rho[3, 3])/2 - 0.5*cmath.sqrt((rho[0, 0] - rho[3, 3])**2 + 4*abs(rho[0, 3])**2).real
    l5 = 1 + np.sqrt((rho[0, 0]+rho[1, 1]-rho[2, 2]-rho[3, 3])**2 + 4*abs(rho[0, 3])**2)
    l6 = 1 - np.sqrt((rho[0, 0]+rho[1, 1]-rho[2, 2]-rho[3, 3])**2 + 4*abs(rho[0, 3])**2)
    if (rho[0, 0]>0 and rho[3, 3]>0 and l1>0 and l2>0):
        discord_term1 = -(rho[0, 0]+rho[2, 2])*math.log(rho[0, 0]+rho[2, 2])/math.log(2) - (rho[1, 1]+rho[3, 3])*math.log(rho[1, 1]+rho[3, 3])/math.log(2)
        discord_term2 = rho[1, 1]*math.log(rho[1, 1])/math.log(2) + rho[2, 2]*math.log(rho[2, 2])/math.log(2) + l1*cmath.log(l1)/math.log(2) + l2*math.log(l2)/math.log(2)
        discord_term3 = -(1/2)*l5*math.log(l5/2)/math.log(2) - (1/2)*l6*math.log(l6/2)/math.log(2)
        discord[i] = discord_term1 + discord_term2 + discord_term3
    
    discord[0] = 1
    


data = np.zeros((len(t), 3))

data[:, 0] = t
data[:, 1] = concur
data[:, 2] = discord

np.savetxt("concur_discord_2_qubit_global_int_on", data)