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

model_file = 'moon_1.dat'
V0_file = 'tidal_potential.dat'
litho_file = 'litho_thickness.dat'

time = 1200          # Maxwell time

# tidal or surface loading
Flag_tidal = 1
Flag_surface_loading = 0

def grav_acc(r,rhoc,rhom):
    n = len(r)
    c = 4*pi*G/3
    mass = np.zeros((n,))
    for l in range(n):
        if l == 0:
            mass[l] = rhoc*r[l]**3
        else:
            mass[l] = mass[l-1] + rhom*(r[l]**3-r[l-1]**3)
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

# interpolate potential or lithosphere radius Lith for Maxwell time t
def interpolate_t(t,V):
    n = V.shape[1] - 1
    for i in range(n):
        if t >= V[0,i]:
            if t < V[0,i+1]:
                return V[1,i] + (V[1,i+1]-V[1,i])/(V[0,i+1]-V[0,i])*(t-V[0,i])
            else:
                continue       

# update radius profile for lithoshpere thickening process
def litho_update(v,r,r0):
    n = len(r)-1
    v_new = np.zeros(n)
    for iv in range(n):
        for ir in range(len(v)):
            if r[iv+1] <= r0[ir+1]:
                v_new[iv] = v[ir]
                break
            else:
                continue
    return v_new
            
def model_readin(file_name):
    fid = open(file_name,'r')
    model = []
    fid.readline()      # skip header      
    for line in fid.readlines():
        model.append(line.rstrip('\n').split())
    model = np.array(model,dtype=np.float32)
    fid.close()
    return model

    
# read in time evolving potential and viscosity structure
V = model_readin(V0_file).T
if Flag_surface_loading == 1:     # if surface loading
    V[1] /= (2*l+1)
    
# read in time dependent lithosphere thickness
Lith = model_readin(litho_file).T

# model setup, before considering lithosphere thickening
model_0 = model_readin(model_file)
N0 = len(model_0)-1


r0 = model_0[:,0]
R0 = r0[-1]
eta0 = model_0[1:,1]
rho0 = model_0[:,2]
rhoc,rhom = rho0[0],rho0[1]
mu0 = model_0[1:,3]

G = 6.67e-11
rsg = 4*pi*G*rhom**2*R0**2/mu0[0]
q0 = 4*pi*G*rhom*R0
tau0 = eta0/mu0              # Maxwell time for each layer
tau0_yr = tau0/3600/24/365    # Maxwell time in yrs

# time evolution, non-dim values 
#tau_ref = tau0.min()
tau_ref = tau0[0]
tau_0 = tau0[0]/tau_ref
#dt = tau.min()       # each time step equals Maxwell time
prefactor = 1.0       # dt = prefactor*tau
dt = prefactor*tau_0
step = int(time/dt) + 1   # total time steps

# refine lithospheric layer for each time step, according to formulation
r_lith = np.zeros(step)
for it in range(step):
    r_lith[it] = R0 - interpolate_t(it*dt,Lith)*1000

if np.unique(r_lith).size == 1:
    fixed_lith = 1
else:
    fixed_lith = 0
# combine r0 and r_lith
r1 = np.sort(np.unique(np.concatenate((r0,r_lith))))     
N = len(r1)
#g0 = np.array([9.8]*(N+1))
g0 = grav_acc(r1,rhoc,rhom)     # compute gravitational acceleration
r = r1/R0
rb,rs = r[0],r[-1]
v = np.log(r)
eta1 = eta0/eta0[0]
mu1 = mu0/mu0[0]
tau1 = tau0/tau_ref
rho = rho0/rhom
d_rho = abs(np.append((rho[:-1]-rho[1:]),rho[-1]))
drho_b,drho_s = d_rho[0],d_rho[-1]   # note we only consider density interface at surface and CMB
g = g0/q0
gb,gs = g[0],g[-1]

# refined layer properties at t=0 for latter use.
eta_lith,tau_lith = eta1[-1],tau1[-1]
mu = litho_update(mu1,r1,r0)
eta = litho_update(eta1,r1,r0)
tau = eta/mu

# time dependent topo and potential at layer interfaces
ur = np.zeros((N,step))
phi = np.zeros((N,step))
# traction components at internal layer interfaces
if N > 2:
    trr_m = np.zeros((N-2,step))
    trt_m = np.zeros((N-2,step))

for it in range(step):
    if it%100 == 0:
        print("time step {0:d}".format(it))
    # interpolate potential Vn, dVn, rln, for current time
    Vn = interpolate_t(it*dt,V)             # Vn, current time step
    if fixed_lith == 0 and it > 0:    
        rln = r_lith[it]                        # rln, current radius of lithosphere bottom
        if rln < R0:                        # if rln == R0, lith thickening not starting yet
            ind = (r1[:-1] == rln) 
    if it == 0:
        eta_bar = mu   # t=0
        Vp,V0 = Vn,Vn                       # Vp, previous time step
        dVn = Vp                            # dVn, change of Vn in dt        
    elif it > 0:
        if fixed_lith == 0 and rln < R0:
            # update eta,tau,beta,alpha
            eta[ind] = eta_lith
            tau[ind] = tau_lith
        beta = dt/(dt+tau)
        alpha = 1 - beta
        # beta, alpha at cmb and surface
        beta_b,beta_s = beta[0],beta[-1]
        alpha_b,alpha_s = alpha[0],alpha[-1]
        eta_bar = eta/(tau+dt)   # t>0    
        dVn = Vn - Vp
            
    # build A and P matrices for current time step 
    if it == 0 or it == 1:
        P1 = np.array([np.eye(4)]*(N-1))
        P2 = np.array([np.eye(4)]*(N-1))
        for layer in range(N-1):
            A = matrix_a(eta_bar[layer])
            P1[layer] = prop_matrix(A,v[layer],v[layer+1])
    # if it > 1, only update P1 in one layer    
    elif it > 1 and fixed_lith == 0 and rln < R0:
        A = matrix_a(eta_bar[ind])
        # propagator matrix for layer that turns into lithosphere
        P1[ind] = prop_matrix(A,v[:-1][ind],v[1:][ind])  
        
    # P2(rk->rs),P2(rk-1 -> rs),...,P2(rb->rs)
    # for computational efficiency...
    if it == 0 or it == 1:
        l_init = 0
    else:
        l_init = it-1
    if it <= 1 or (fixed_lith == 0 and rln < R0):
        for layer in range(l_init,N-1):
            if layer == 0:
                P2[layer] = P1[-1]
            else:
                P2[layer] = np.dot(P2[layer-1],P1[N-2-layer])

    # build a, b coefficients, which relate to solutions at rb,rs
    if it == 0:
        # linear equations coeffs, update a3,a5,b3,b5 only for each time step.
        # these coeffs are only associated with values at rb and rs
        ca = np.zeros(5,dtype=np.float32)
        cb = np.zeros(6,dtype=np.float32)
        ca[0] = -rsg*drho_b**2*rb**2/(2*l+1)
        ca[1] = -rsg*drho_b*drho_s*rb**(l+1)/(2*l+1)
        ca[3] = rsg*drho_b*rb*gb
        ca[4] = 0                  # update every time step 
        cb[0] = rsg*drho_s*drho_b*rb**(l+2)/(2*l+1)
        cb[1] = rsg*drho_s**2/(2*l+1)
        cb[3] = -rsg*drho_s*gs   
        cb[4] = 0       # update every time step      
        if Flag_surface_loading == 1:
            Sn = Vn*(2*l+1)
            cb[5] = -rsg*rs*Sn*gs          # update if area density Sn changes with time
             
    ca[2] = -rsg*drho_b*rb**(l+1)*dVn       # update each time step
    cb[2] = rsg*drho_s*dVn                  # update each time step
    # discontinuity cumulatives
    DIS = np.zeros((4,1),dtype=np.float32)    # needs to be updated each time step if N > 2
    if N > 2:
        # cc vectors due to discontinuities
        cc =  np.zeros((4,N-2),dtype=np.float32)  

    if it > 0:
        ca[4] = -rb*rsg*beta_b*drho_b*(phi[0,it-1]+rb**l*Vp-gb*ur[0,it-1])
        cb[4] = rs*rsg*beta_s*drho_s*(phi[-1,it-1]+rs**l*Vp-gs*ur[-1,it-1])
        # deal with discontinuities DIS when t>0...
        if N > 2: 
            # update cc and DIS
            for k in range(1,N-1):
                # note that we already presume mantle density rho[k] is constant
                cc[2,k-1] = r[k]*(alpha[k-1]-alpha[k])*(trr_m[k-1,it-1]+rsg*rho[-1]*(phi[k,it-1]+r[k]**l*Vp-g[k]*ur[k,it-1]))
                cc[3,k-1] = r[k]*(alpha[k-1]-alpha[k])*trt_m[k-1,it-1]   
                DIS += np.dot(P2[N-2-k],cc[:,k-1].reshape((-1,1)))          
        if Flag_surface_loading == 1:
            Sn = (Vn - alpha_s*Vp)*(2*l+1)
            cb[5] = -rsg*rs*Sn*gs
               
    C = linear_eqn_n(P2[-1],DIS,ca,cb)
    
    Y = solve(C[0],C[1]).reshape((4,))    # solution vector
    
    d_ur_s = Y[0]
    d_ur_b = Y[2]
    d_phi_s = (rb**(l+2)*drho_b*Y[2]+rs**l*drho_s*Y[0])/(2*l+1)
    d_phi_b = (rb*drho_b*Y[2]+rb**l*drho_s*Y[0])/(2*l+1)
    
    # obtain solutions at layer interface rm's if N > 1
    # for now, since lithosphere is thickening with time, we obtain solutions at the changing
    # bottom of lithosphere to deal with continuity for the next time step
    if N > 2:
        rY3_b = (ca[0]+ca[3])*Y[2]+ca[1]*Y[0]+(ca[2]+ca[4])
        X_b = np.array([Y[2],Y[3],rY3_b,0]).reshape((4,1))
        Sol_m = np.zeros((N-2,4))
        for k in range(N-2):
            if k == 0:
                X_m = np.dot(P1[k],X_b)
            elif k > 0:
                X_m = np.dot(P1[k],(X_m + cc[:,k-1].reshape((-1,1))))
            d_ur_m = X_m[0]
            d_phi_m = (rb**(l+2)/r[k+1]**(l+1)*drho_b*d_ur_b+r[k+1]**l*drho_s*d_ur_s)/(2*l+1)
            d_trr_m = X_m[2]/r[k+1]-rsg*rho[-1]*(d_phi_m+r[k+1]**l*dVn-g[k+1]*d_ur_m)
            d_trt_m = X_m[3]/r[k+1] 
            Sol_m[k,:] = np.array([d_ur_m,d_phi_m,d_trr_m,d_trt_m])
           
    if it == 0:
        ur[0,it] = d_ur_b
        ur[-1,it] = d_ur_s
        phi[0,it] = d_phi_b
        phi[-1,it] = d_phi_s
        if N > 2:
            ur[1:-1,it] = Sol_m[:,0]
            phi[1:-1,it] = Sol_m[:,1]
            trr_m[:,it] = Sol_m[:,2]
            trt_m[:,it] = Sol_m[:,3]
    else:
        ur[0,it] = ur[0,it-1] + d_ur_b
        ur[-1,it] = ur[-1,it-1] + d_ur_s
        phi[0,it] = phi[0,it-1] + d_phi_b
        phi[-1,it] = phi[-1,it-1] + d_phi_s
        if N > 2:
            ur[1:-1,it] = ur[1:-1,it-1] + Sol_m[:,0]
            phi[1:-1,it] = phi[1:-1,it-1] + Sol_m[:,1]
            Sol_m[:,2] -= beta[:-1]*rsg*rho[-1]*(phi[1:-1,it-1]+r[1:-1]**l*Vp-g[1:-1]*ur[1:-1,it-1])
            trr_m[:,it] = alpha[:-1]*trr_m[:,it-1] + Sol_m[:,2]
            trt_m[:,it] = alpha[:-1]*trt_m[:,it-1] + Sol_m[:,3]

    # update Vp for next time step
    Vp = Vn
# end of loop for time
    
# compute Love numbers at surface
# (to compare with Zhong 2003, surface load has same density as mantle, H0 is load height)
   
k = phi[-1,:]/(rs**l*V0)             
if Flag_surface_loading == 1:
    S0 = V0*(2*l+1)
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





