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

# two-layer viscosity model
N = 2
#eta0 = np.array([1e21,1e21])       # [eta_mantle,eta_litho]
#mu0 = np.array([6.5e10,6.5e10])    # shear modulus
#rhom,rhoc = np.array([3400,7800])         # mantle density
#r0 = np.array([3.4e5,1.68e6,1.74e6])   # layer interface radii

# Zhong etal 2003
eta0 = np.array([1e21,1e30])       # [eta_mantle,eta_litho]
mu0 = np.array([1.4305e11,1.4305e11])    # shear modulus
rhom,rhoc = np.array([5500,10925])         # mantle density
r0 = np.array([3.5035e6,6.27e6,6.37e6])   # layer interface radii
R0 = r0[2]

G = 6.67e-11
tau0 = eta0/mu0              # Maxwell time for each layer
tau0_yr = tau0[0]/3600/24/365    # lower layer Maxwell time in yrs
gb = 4*pi*G/3*rhoc*r0[0]
gm = 4*pi*G/3*(rhoc*r0[0]**3+rhom*(r0[1]**3-r0[0]**3))/r0[1]**2
gs = 4*pi*G/3*(rhoc*r0[0]**3+rhom*(r0[2]**3-r0[0]**3))/r0[2]**2
#gb,gm,gs = 9.8,9.8,9.8

# non-dim values 
rb,rm,rs = r0/R0
v = np.log((rb,rm,rs))
eta = eta0/eta0[0]
mu = mu0/mu0[0]
rsg = 4*pi*G*rhom**2*R0**2/mu0[0]
drho_b = (rhoc-rhom)/rhom
drho_s = rhom/rhom
q0 = 4*pi*G*rhom*R0
gb = gb/q0
gm = gm/q0
gs = gs/q0
# tidal potential magnitude and area density of surface load
T0,S0 = 1.0,1.0
if Flag_tidal == 1:
    V0 = T0      # tidal potential
if Flag_surface_loading == 1:
    V0 = S0/(2*l+1)     # area mass density

# time evolution, non-dim
tau = tau0/tau0[0]
dt = tau[0]       # each time step equals Maxwell time of lower layer
beta_b,beta_s = dt/(dt+tau)      # beta is layer-dependent
alpha_b,alpha_s = tau/(dt+tau)
time = 20        # Maxwell time
step = int(time/dt) + 1   # time step

# the A matrix in isoviscous layer and its eigenvalues
def matrix_a(eta,LL=L):
    A = np.array([[-2,LL,0,0],
              [-1,1,0,1/eta],
              [12*eta,-6*LL*eta,1,LL],
              [-6*eta,2*(2*LL-1)*eta,-1,-2]])
    return A         
    
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

# for two layer model only
cc = [0]*2   # t = 0; needs to be updated each time step

def linear_eqn(P,ca,cb):
    a = np.array([[1-ca[1]*P[0,2],0,-(P[0,0]+P[0,2]*(ca[0]+ca[3])),-P[0,1]],
               [-ca[1]*P[1,2],1,-(P[1,0]+P[1,2]*(ca[0]+ca[3])),-P[1,1]],
               [cb[1]+cb[3]-P[2,2]*ca[1],0,cb[0]-(P[2,0]+P[2,2]*(ca[0]+ca[3])),-P[2,1]],
               [-ca[1]*P[3,2],0,-(P[3,0]+P[3,2]*(ca[0]+ca[3])),-P[3,1]]])
               
    b = np.array([[P[0,2]*(ca[2]+ca[4])],
               [P[1,2]*(ca[2]+ca[4])],
               [P[2,2]*(ca[2]+ca[4])-(cb[2]+cb[4])],
               [P[3,2]*(ca[2]+ca[4])]])
               
    return [a,b]
    
# this is for two layer model..
def linear_eqn_2(P,Q,ca,cb,cc):
    a = np.array([[1-ca[1]*P[0,2],0,-(P[0,0]+P[0,2]*(ca[0]+ca[3])),-P[0,1]],
               [-ca[1]*P[1,2],1,-(P[1,0]+P[1,2]*(ca[0]+ca[3])),-P[1,1]],
               [cb[1]+cb[3]-P[2,2]*ca[1],0,cb[0]-(P[2,0]+P[2,2]*(ca[0]+ca[3])),-P[2,1]],
               [-ca[1]*P[3,2],0,-(P[3,0]+P[3,2]*(ca[0]+ca[3])),-P[3,1]]])
               
    b = np.array([[P[0,2]*(ca[2]+ca[4])+(Q[0,2]*cc[0]+Q[0,3]*cc[1])],
               [P[1,2]*(ca[2]+ca[4])+(Q[1,2]*cc[0]+Q[1,3]*cc[1])],
               [P[2,2]*(ca[2]+ca[4])-(cb[2]+cb[4]+cb[5])+(Q[2,2]*cc[0]+Q[2,3]*cc[1])],
               [P[3,2]*(ca[2]+ca[4])+(Q[3,2]*cc[0]+Q[3,3]*cc[1])]])
               
    return [a,b]

# time dependent topo and potential at surface and cmb 
ur_s = np.zeros(step)
phi_s = np.zeros(step)
ur_b = np.zeros(step)
phi_b = np.zeros(step)
ur_m = np.zeros(step)
phi_m = np.zeros(step)
# at layer interface rm
trr_m = np.zeros(step)  # traction component at rm (may change) 
trt_m = np.zeros(step)

for it in range(step):
    if it%1e5 == 0:
        print("time step {0:d}".format(it))
    # build A and P matrices for t=0 and t>0
    if it == 0:
        eta_bar = mu   # t=0
    elif it == 1:
        eta_bar = eta/(tau+dt)   # t>0
    if it == 0 or it == 1:
        P = [np.eye(4)]*3
        for layer in range(N):
            A = matrix_a(eta_bar[layer])
            P[layer] = prop_matrix(A,v[layer],v[layer+1])
        P[2] = np.dot(P[1],P[0])   # P = [P(rb->rm),P(rm->rs),P(rb->rs)]
    if it > 0:
        ca[2] = 0
        cb[2] = 0
        ca[4] = -rb*rsg*beta_b*drho_b*(phi_b[it-1]+rb**l*V0-gb*ur_b[it-1])
        cb[4] = rs*rsg*beta_s*drho_s*(phi_s[it-1]+rs**l*V0-gs*ur_s[it-1])
        cc[0] = rm*(alpha_b-alpha_s)*(trr_m[it-1]+rsg*drho_s*(phi_m[it-1]+rm**l*V0-gm*ur_m[it-1]))
        cc[1] = rm*(alpha_b-alpha_s)*trt_m[it-1]        
        if Flag_surface_loading == 1:
            cb[5] = -rsg*rs*beta_s*S0*gs
               
#    CC = linear_eqn(P[2],ca,cb)
    C = linear_eqn_2(P[2],P[1],ca,cb,cc)
    
    Y = solve(C[0],C[1]).reshape((4,))    # solution vector
    
    d_ur_s = Y[0]
    d_ur_b = Y[2]
    d_phi_s = (rb**(l+2)*drho_b*Y[2]+rs**l*drho_s*Y[0])/(2*l+1)
    d_phi_b = (rb*drho_b*Y[2]+rb**l*drho_s*Y[0])/(2*l+1)
    
    # obtain solutions at layer interface rm...
    rY3_b = (ca[0]+ca[3])*Y[2]+ca[1]*Y[0]+(ca[2]+ca[4])
    X_b = np.array([Y[2],Y[3],rY3_b,0]).reshape((4,1))
    X_m = np.dot(P[0],X_b)
    d_ur_m = X_m[0]
    d_phi_m = (rb**(l+2)/rm**(l+1)*drho_b*d_ur_b+rm**l*drho_s*d_ur_s)/(2*l+1)
    d_trr_m = X_m[2]/rm-rsg*drho_s*(d_phi_m-gm*d_ur_m)
    d_trt_m = X_m[3]/rm
    
    if it == 0:
        ur_s[it] = d_ur_s
        ur_b[it] = d_ur_b
        ur_m[it] = d_ur_m
        phi_s[it] = d_phi_s
        phi_b[it] = d_phi_b
        phi_m[it] = d_phi_m
        d_trr_m -= rsg*drho_s*rm**l*V0
        trr_m[it] = d_trr_m
        trt_m[it] = d_trt_m
    else:
        ur_s[it] = ur_s[it-1] + d_ur_s
        ur_b[it] = ur_b[it-1] + d_ur_b
        ur_m[it] = ur_m[it-1] + d_ur_m
        phi_s[it] = phi_s[it-1] + d_phi_s
        phi_b[it] = phi_b[it-1] + d_phi_b
        phi_m[it] = phi_m[it-1] + d_phi_m
        d_trr_m -= beta_b*rsg*drho_s*(phi_m[it-1]+rm**l*V0-gm*ur_m[it-1])
        trr_m[it] = alpha_b*trr_m[it-1] + d_trr_m
        trt_m[it] = alpha_b*trt_m[it-1] + d_trt_m


# compute Love numbers at surface
# (to compare with Zhong 2003, surface load has same density as mantle, H0 is load height)
#H0 = S0/drho_s    
k = phi_s/(rs**l*V0)
h = ur_s*gs/(rs**l*V0)    # load Love number notation
#h = ur_s/H0                 # Zhong 2003 notation
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



