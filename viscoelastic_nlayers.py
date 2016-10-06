# viscoelastic relaxation calculation prototype

import numpy as np
from math import pi
from numpy.linalg import solve
#from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd


# harmonic degree of forcing
l = 2
L = l*(l+1)

# tidal or surface loading
Flag_tidal = 1
Flag_surface_loading = 0

def grav_acc(r,rho):
    n = len(r)
    c = 4*pi*G/3
    mass = np.zeros((n,))
    for l in range(n):
        if l == 0:
            mass[l] = rho[l]*r[l]**3
        else:
            mass[l] = mass[l-1] + rho[l]*(r[l]**3-r[l-1]**3)
    g = c*mass/r**2
    return g

# the A matrix in isoviscous layer and its eigenvalues
def matrix_a(eta,LL=L):
    A = np.array([[-2,LL,0,0],
              [-1,1,0,1/eta],
              [12*eta,-6*LL*eta,1,LL],
              [-6*eta,2*(2*LL-1)*eta,-1,-2]])
    return A         

# eigenvalues of A matrix    
lambdas = (l+1,-l,l-1,-l-2)

# propagator matrix from v1 to v2
def prop_matrix(A,v1,v2,ld=lambdas):
    P = np.zeros((4,4),dtype=np.float)
    for i in ld:
        c = np.exp(i*(v2-v1))
        p = np.eye(4,dtype=np.float)
        for j in ld:
            if j != i:
                p = np.dot(p,(A-np.eye(4)*j)/(i-j))
        P = P + c*p       
    # or...
    # P = expm(A*(v2-v1))
    return P

# generalized to N layer model...
def linear_eqn_n(P,DIS,ca,cb):
    a = np.array([[1-ca[1]*P[0,2],0,-(P[0,0]+P[0,2]*(ca[0]+ca[3])),-P[0,1]],
               [-ca[1]*P[1,2],1,-(P[1,0]+P[1,2]*(ca[0]+ca[3])),-P[1,1]],
               [cb[1]+cb[3]-P[2,2]*ca[1],0,cb[0]-(P[2,0]+P[2,2]*(ca[0]+ca[3])),-P[2,1]],
               [-ca[1]*P[3,2],0,-(P[3,0]+P[3,2]*(ca[0]+ca[3])),-P[3,1]]])
               
    b = np.array([[P[0,2]*(ca[2]+ca[4])+DIS[0,0]],
               [P[1,2]*(ca[2]+ca[4])+DIS[1,0]],
               [P[2,2]*(ca[2]+ca[4])-(cb[2]+cb[4]+cb[5])+DIS[2,0]],
               [P[3,2]*(ca[2]+ca[4])+DIS[3,0]]])
               
    return [a,b]


# tidal potential magnitude and area density of surface load
T0,S0 = 1.0,1.0
if Flag_tidal == 1:
    V0 = T0      # tidal potential
if Flag_surface_loading == 1:
    V0 = S0/(2*l+1)     # area mass density

# model setup
model_file = 'moon_1.dat'
fid = open(model_file,'r')
N = -2
model = []
for line in fid.readlines():
    if N == -2:
        N = -1
        continue
    ln = line.rstrip('\n').split()
    model.append(ln)
    N += 1
model = np.array(model,dtype=np.float32)
r0 = model[:,0]
R0 = r0[-1]
eta0 = model[1:,1]
rho0 = model[:,2]
rhom = rho0[1]
mu0 = model[1:,3]

G = 6.67e-11
tau0 = eta0/mu0              # Maxwell time for each layer
tau0_yr = tau0/3600/24/365    # Maxwell time in yrs
g0 = grav_acc(r0,rho0)       # compute gravitational acceleration
#g0 = np.array([9.8]*(N+1))

# non-dim values 
r = r0/R0
rb,rs = r[0],r[-1]
v = np.log(r)
eta = eta0/eta0[0]
mu = mu0/mu0[0]
rsg = 4*pi*G*rhom**2*R0**2/mu0[0]
rho = rho0/rhom
d_rho = abs(np.append((rho[:-1]-rho[1:]),rho[-1]))
# note we only consider density interface at surface and CMB
drho_b,drho_s = d_rho[0],d_rho[-1]
q0 = 4*pi*G*rhom*R0
g = g0/q0
gb,gs = g[0],g[-1]

# time evolution, non-dim
#tau_ref = tau0.min()
tau_ref = tau0[0]
tau = tau0/tau_ref
#dt = tau.min()       # each time step equals Maxwell time
dt = 1.0
beta = dt/(dt+tau)
alpha= 1 - beta
beta_b,beta_s = beta[0],beta[-1]
alpha_b,alpha_s = alpha[0],alpha[-1]
time = 10000           # Maxwell time
step = int(time/dt) + 1   # time step

   
# linear equations coeffs, update a3,a5,b3,b5 only for each time step.
# these coeffs are only associated with values at rb and rs
ca = [0]*5
cb = [0]*6
# at t = 0
ca[0] = -rsg*drho_b**2*rb**2/(2*l+1)
ca[1] = -rsg*drho_b*drho_s*rb**(l+1)/(2*l+1)
ca[2] = -rsg*drho_b*rb**(l+1)*V0       # update if V0 changes with time
ca[3] = rsg*drho_b*rb*gb
ca[4] = 0       # update every time step 

cb[0] = rsg*drho_s*drho_b*rb**(l+2)/(2*l+1)
cb[1] = rsg*drho_s**2/(2*l+1)
cb[2] = rsg*drho_s*V0                  # update if V0 changes with time
cb[3] = -rsg*drho_s*gs
cb[4] = 0       # update every time step 
if Flag_surface_loading == 1:
    cb[5] = -rsg*rs*S0*gs          # update if S0 changes with time

# discontinuity
DIS = np.zeros((4,1))   # t = 0; needs to be updated each time step

# time dependent topo and potential at layer interfaces
ur = np.zeros((N+1,step))
phi = np.zeros((N+1,step))
# stress at layer interface rm's
trr_m = np.zeros((N-1,step))  # traction component at rm (may change) 
trt_m = np.zeros((N-1,step))

for it in range(step):
    if it%1000 == 0:
        print("time step {0:d}".format(it))
    # build A and P matrices for t=0 and t>0
    if it == 0:
        eta_bar = mu   # t=0
    elif it == 1:
        eta_bar = eta/(tau+dt)   # t>0
    if it == 0 or it == 1:
        P1 = [np.eye(4)]*N
        P2 = [np.eye(4)]*N
        for layer in range(N):
            A = matrix_a(eta_bar[layer])
            P1[layer] = prop_matrix(A,v[layer],v[layer+1])
        # P2(rk->rs),P2(rk-1 -> rs),...,P2(rb->rs)
        for layer in range(N):
            if layer == 0:
                P2[layer] = P1[-1]
            else:
                P2[layer] = np.dot(P2[layer-1],P1[N-1-layer])
        # P(rb->r1),P1(rb->r2),...,P1(rb->rs)
        # for layer in range(N):
        #    if layer > 0:
        #        P1[layer] = np.dot(P1[layer],P1[layer-1])
                
    if it > 0:
        ca[2] = 0
        cb[2] = 0
        ca[4] = -rb*rsg*beta_b*drho_b*(phi[0,it-1]+rb**l*V0-gb*ur[0,it-1])
        cb[4] = rs*rsg*beta_s*drho_s*(phi[-1,it-1]+rs**l*V0-gs*ur[-1,it-1])
        # deal with discontinuities DIS when t>0...
        if N > 1:
            DIS = np.zeros((4,1))
            cc =  np.zeros((4,N-1))   # cc vectors due to discontinuities
            for k in range(1,N):
                # note that we already presume mantle density rho[k] is constant
                cc[2,k-1] = r[k]*(alpha[k-1]-alpha[k])*(trr_m[k-1,it-1]+rsg*rho[k]*(phi[k,it-1]+r[k]**l*V0-g[k]*ur[k,it-1]))
                cc[3,k-1] = r[k]*(alpha[k-1]-alpha[k])*trt_m[k-1,it-1]   
                DIS += np.dot(P2[N-1-k],cc[:,k-1].reshape((-1,1)))
            
        if Flag_surface_loading == 1:
            cb[5] = -rsg*rs*beta_s*S0*gs
    elif it == 0:
        cc = np.zeros((4,N-1)) 
            
               
#    CC = linear_eqn(P[2],ca,cb)
#    C = linear_eqn_2(P[2],P[1],ca,cb,cc)
    C = linear_eqn_n(P2[-1],DIS,ca,cb)
    
    Y = solve(C[0],C[1]).reshape((4,))    # solution vector
    
    d_ur_s = Y[0]
    d_ur_b = Y[2]
    d_phi_s = (rb**(l+2)*drho_b*Y[2]+rs**l*drho_s*Y[0])/(2*l+1)
    d_phi_b = (rb*drho_b*Y[2]+rb**l*drho_s*Y[0])/(2*l+1)
    
    # obtain solutions at layer interface rm's if N > 1
    if N > 1:
        rY3_b = (ca[0]+ca[3])*Y[2]+ca[1]*Y[0]+(ca[2]+ca[4])
        X_b = np.array([Y[2],Y[3],rY3_b,0]).reshape((4,1))
        Sol_m = np.zeros((N-1,4))
        for k in range(N-1):
            if k == 0:
                X_m = np.dot(P1[k],X_b)
            elif k > 0:
                X_m = np.dot(P1[k],(X_m + cc[:,k-1].reshape((-1,1))))
            d_ur_m = X_m[0]
            d_phi_m = (rb**(l+2)/r[k+1]**(l+1)*drho_b*d_ur_b+r[k+1]**l*drho_s*d_ur_s)/(2*l+1)
            d_trr_m = X_m[2]/r[k+1]-rsg*rho[k+1]*(d_phi_m-g[k+1]*d_ur_m)
            d_trt_m = X_m[3]/r[k+1] 
            Sol_m[k,:] = np.array([d_ur_m,d_phi_m,d_trr_m,d_trt_m])
           
    if it == 0:
        ur[0,it] = d_ur_b
        ur[-1,it] = d_ur_s
        phi[0,it] = d_phi_b
        phi[-1,it] = d_phi_s
        if N > 1:
            ur[1:-1,it] = Sol_m[:,0]
            phi[1:-1,it] = Sol_m[:,1]
            trr_m[:,it] = Sol_m[:,2] - (rsg*rho[1:-1]*r[1:-1]**l*V0)
            trt_m[:,it] = Sol_m[:,3]
    else:
        ur[0,it] = ur[0,it-1] + d_ur_b
        ur[-1,it] = ur[-1,it-1] + d_ur_s
        phi[0,it] = phi[0,it-1] + d_phi_b
        phi[-1,it] = phi[-1,it-1] + d_phi_s
        if N > 1:
            ur[1:-1,it] = ur[1:-1,it-1] + Sol_m[:,0]
            phi[1:-1,it] = phi[1:-1,it-1] + Sol_m[:,1]
            Sol_m[:,2] -= beta[:-1]*rsg*rho[1:-1]*(phi[1:-1,it-1]+r[1:-1]**l*V0-g[1:-1]*ur[1:-1,it-1])
            trr_m[:,it] = alpha[:-1]*trr_m[:,it-1] + Sol_m[:,2]
            trt_m[:,it] = alpha[:-1]*trt_m[:,it-1] + Sol_m[:,3]


# compute Love numbers at surface
# (to compare with Zhong 2003, surface load has same density as mantle, H0 is load height)
   
k = phi[-1,:]/(rs**l*V0)             
if Flag_surface_loading == 1:
    H0 = S0/drho_s 
    h = ur[-1,:]/H0         # Zhong 2003 notation
if Flag_tidal == 1:
    h = ur[-1,:]*gs/(rs**l*V0)    # load Love number notation
# l = Y[1]*gs/(rs**l*V0)
#output = "k={0:f}, h={1:f}, l={2:f}".format(k,h,l)

print(k[-1])
print(h[-1])

df1 = pd.read_table("casea.pttl_time.dat",delimiter=' ',names=['time','response','whatever'])
df2 = pd.read_table("casea.tps_time.dat",delimiter=' ',names=['time','response','whatever'])
df3 = pd.read_table("caseb.pttl_time.dat",delimiter=' ',names=['time','response','whatever'])
df4 = pd.read_table("caseb.tps_time.dat",delimiter=' ',names=['time','response','whatever'])


plt.plot(np.array(range(step))*dt,k,'b',np.array(range(step))*dt,h,'r')
#plt.plot(df1['time'],df1['response']-1.0,'y--',df2['time'],df2['response'],'y--')
#plt.plot(df3['time'],df3['response']-1.0,'k--',df4['time'],df4['response'],'k--')

plt.show()



