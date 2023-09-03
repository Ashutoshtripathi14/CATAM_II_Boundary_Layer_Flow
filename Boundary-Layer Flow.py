#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve


# In[2]:


#---start of def---
def falkner_skan_setup(f,nu,m):
    dfdnu=[0,0,0]
    dfdnu[0]=f[1]
    dfdnu[1]=f[2]
    dfdnu[2]=m*(f[1]**2)-(1/2)*(m+1)*f[0]*f[2]-m
    return dfdnu
#---end of def---
#---start of def---
def falkner_skan_integrator_type1(m,S,nuspan,reltol,abstol):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup, f_init, nuspan, args=(m,),rtol=reltol,atol=abstol)
    return sol
#---end of def---
#---start of def---
def falkner_skan_shooting_method_objective(S,m,nuspan,reltol=1e-8,abstol=1e-8):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup, f_init, nuspan, args=(m,),rtol=reltol,atol=abstol)
    df= sol[:,1]
    return df[-1]-1
#---end of def---
#---start of def---
def falkner_skan_shooting_method(m,initval):
    Sm, = fsolve(falkner_skan_shooting_method_objective, initval, args=(m,np.linspace(0,20,501),),xtol=1e-8)
    return Sm
# np.info(scipy.optimize.fsolve)
#---end of def---
#---start of def---  
def falkner_skan_shooting_method_plot(m,initval):
    S_ideal=falkner_skan_shooting_method(m,initval);
    print('S_ideal',S_ideal)
    sol=falkner_skan_integrator_type1(m,S_ideal,np.linspace(0,10,501),1e-8,1e-8);
    print(sol[:,1][-1]-1);
    plt.plot(np.linspace(0,10,501), sol[:, 1], 'b', label='theta(t)')
#---end of def---
#---start of def---
def falkner_skan_shooting_method_objective2(S,m,nuspan,reltol=1e-8,abstol=1e-8):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup, f_init, nuspan, args=(m,),rtol=reltol,atol=abstol)
    df= sol[:,1]
    if(S>0):
        return 100
    return df[-1]-1
#---end of def---
#---start of def---
def falkner_skan_shooting_method2(m,initval):
    Sm, = fsolve(falkner_skan_shooting_method_objective2, initval, args=(m,np.linspace(0,40,501),),xtol=1e-8)
    return Sm
# np.info(scipy.optimize.fsolve)
#---end of def---
#---start of def---  
def falkner_skan_shooting_method_plot2(m,initval):
    S_ideal=falkner_skan_shooting_method2(m,initval);
    print('S_ideal',S_ideal)
    sol=falkner_skan_integrator_type1(m,S_ideal,np.linspace(0,40,501),1e-8,1e-8);
    print(sol[:,1][-1]-1);
    plt.plot(np.linspace(0,40,501), sol[:, 1], 'b', label='theta(t)')
#---end of def---
#---start of def---
def falkner_skan_setup3(f,nu,beta):
    dfdnu=[0,0,0]
    dfdnu[0]=f[1]
    dfdnu[1]=f[2]
    dfdnu[2]=beta*(f[1]**2-1)-f[0]*f[2]
    return dfdnu
#---end of def---
#---start of def---
def falkner_skan_integrator_type3(beta,S,nuspan,reltol,abstol):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup3, f_init, nuspan, args=(beta,),rtol=reltol,atol=abstol)
    return sol
#---end of def---
#---start of def---
def falkner_skan_shooting_method_objective3(S,beta,nuspan,reltol=1e-8,abstol=1e-8):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup3, f_init, nuspan, args=(beta,),rtol=reltol,atol=abstol)
    df= sol[:,1]
    return df[-1]-1
#---end of def---
#---start of def---
def falkner_skan_shooting_method3(beta,initval):
    Sm, = fsolve(falkner_skan_shooting_method_objective3, initval, args=(beta,np.linspace(0,20,501),),xtol=1e-8)
    return Sm
# np.info(scipy.optimize.fsolve)
#---end of def---
#---start of def---  
def falkner_skan_shooting_method_plot3(beta,initval):
    S_ideal=falkner_skan_shooting_method3(beta,initval);
    print('S_ideal',S_ideal)
    sol=falkner_skan_integrator_type3(beta,S_ideal,np.linspace(0,40,501),1e-8,1e-8);
    print(sol[:,1][-1]-1);
    plt.plot(np.linspace(0,40,501), sol[:, 1], 'b', label='theta(t)')
#---end of def---


#---start of def---
def falkner_skan_integrator_type4(m,S,nuspan,reltol,abstol):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup, f_init, nuspan, args=(m,),rtol=reltol,atol=abstol)
    return sol
#---end of def---
#---start of def---
def falkner_skan_shooting_method_objective4(S,m,nuspan,reltol=1e-8,abstol=1e-8):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup, f_init, nuspan, args=(m,),rtol=reltol,atol=abstol)
    df= sol[:,1]
    return -df[-1]-1
#---end of def---
#---start of def---
def falkner_skan_shooting_method4(m,initval):
    Sm, = fsolve(falkner_skan_shooting_method_objective4, initval, args=(m,np.linspace(0,20,501),),xtol=1e-8)
    return Sm
# np.info(scipy.optimize.fsolve)
#---end of def---
#---start of def---  
def falkner_skan_shooting_method_plot4(m,initval):
    S_ideal=falkner_skan_shooting_method4(m,initval);
    print('S_ideal',S_ideal)
    sol=falkner_skan_integrator_type4(m,S_ideal,np.linspace(0,20,501),1e-5,1e-5);
    print(sol[:,1][-1]-1);
    plt.plot(np.linspace(0,20,501), sol[:, 1], 'b', label='theta(t)')
#---end of def---
#---start of def---
def falkner_skan_setup5(f,nu,beta):
    dfdnu=[0,0,0]
    dfdnu[0]=f[1]
    dfdnu[1]=f[2]
    dfdnu[2]=beta*(f[1]**2-1)-f[0]*f[2]
    return dfdnu
#---end of def---
#---start of def---
def falkner_skan_integrator_type5(beta,S,nuspan,reltol,abstol):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup5, f_init, nuspan, args=(beta,),rtol=reltol,atol=abstol)
    return sol
#---end of def---
#---start of def---
def falkner_skan_shooting_method_objective5(S,beta,nuspan,reltol=1e-8,abstol=1e-8):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup5, f_init, nuspan, args=(beta,),rtol=reltol,atol=abstol)
    df= sol[:,1]
    if(S>0):
        return 100
    return df[-1]-1
#---end of def---
#---start of def---
def falkner_skan_shooting_method5(beta,initval):
    Sm, = fsolve(falkner_skan_shooting_method_objective5, initval, args=(beta,np.linspace(0,20,501),),xtol=1e-8)
    return Sm
# np.info(scipy.optimize.fsolve)
#---end of def---



# # Question 2

# In[583]:


sol=falkner_skan_integrator_type1(0,1,np.linspace(0,100,501),1e-8,1e-8);
plt.plot(np.linspace(0,100,501), sol[:, 1], 'b', label='theta(t)')
plt.savefig('fig2.eps', format='eps')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta))$')
plt.show()
asym=sol[:,1][-1]


# In[584]:


x=np.linspace(0,20,501)
y=sol[:,1]
for i in range(len(x)):
    print(x[i]," ",y[i])


# In[585]:


print(asym)


# In[586]:


sol=falkner_skan_integrator_type1(0,1,np.linspace(0,5.5,501),1e-5,1e-5);
sol1=sol[:,1]
sol2=np.log(asym-sol1)
x=np.linspace(0,5.5,501)
x1=np.log(x)
plt.plot(x[300:], sol2[300:], 'b', label='theta(t)')
plt.xlabel('$\eta$')
plt.ylabel('$\log{(f\'_0-f}$'+" "+'$\'(\eta))$')
plt.savefig('fig1.eps', format='eps')
plt.show()


# # Question 3

# ## (a) Case of $m=\frac{2}{5}$

# In[587]:


for x in range(70,100,1):
    sol=falkner_skan_integrator_type1(2/5,x/100,np.linspace(0,7,501),1e-8,1e-8)
    plt.plot(np.linspace(0,7,501), sol[:, 1], label='f\'')
    plt.axhline(1,label='Asymptote')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig3.eps', format='eps')
plt.show()
#so suggests Algebraic Divergence + exponential convergence 


# In[588]:


sol=falkner_skan_integrator_type1(2/5,0.4,np.linspace(0,7.4,501),1e-8,1e-8)
sol1=sol[:,1]
sol2=np.log(np.abs(1-sol[:,1]))
x=(np.linspace(0,7.5,501))
x2=np.log(x)
plt.plot(x, sol1, label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig6.eps', format='eps')
plt.show()


# In[589]:


Sval=falkner_skan_shooting_method(2/5,2)
sol=falkner_skan_integrator_type1(2/5,Sval,np.linspace(0,7,501),1e-8,1e-8)
sol1=sol[:,1]
sol2=np.log((1-sol[:,1]))
x=(np.linspace(0,7,501))
x2=np.log(x)
plt.plot(x, sol1, label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig7.eps', format='eps')
plt.show()
plt.plot(x[300:], sol2[300:], label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$\log{(\mid 1-f}$'+" "+'$\'(\eta)\mid)$')
plt.savefig('fig8.eps', format='eps')
plt.show()


# In[590]:


sol=falkner_skan_integrator_type1(2/5,90/100,np.linspace(0,200,501),1e-8,1e-8)
sol1=sol[:,1]
sol2=np.log(np.abs(1-sol[:,1]))
x=(np.linspace(0,200,501))
x2=np.log(x)
plt.plot(x, sol1, label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig9.eps', format='eps')
plt.show()
plt.plot(x2[300:], sol2[300:], label='f\'')
plt.xlabel('$\log{(\eta)}$')
plt.ylabel('$\log{(\mid 1-f}$'+" "+'$\'(\eta)\mid)$')
plt.savefig('fig10.eps', format='eps')
plt.show()


# ## (a) Case of $m=1$

# In[591]:


for x in range(110,140,1):
    sol=falkner_skan_integrator_type1(1,x/100,np.linspace(0,5,501),1e-8,1e-8)
    plt.plot(np.linspace(0,5,501), sol[:, 1], label='f\'')
    plt.axhline(1,label='Asymptote')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig4.eps', format='eps')
plt.show()
#so suggests only exponential divergence + exponential convergence 


# In[593]:


sol=falkner_skan_integrator_type1(1,0.3,np.linspace(0,18,501),1e-12,1e-12)
sol1=sol[:,1]
sol2=np.log(np.abs(1-sol[:,1]))
x=(np.linspace(0,18,501))
x2=np.log(x)
plt.plot(x, sol1, label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig15.eps', format='eps')
plt.show()


# In[594]:


Sval=falkner_skan_shooting_method(1,2)
sol=falkner_skan_integrator_type1(1,Sval,np.linspace(0,5,501),1e-8,1e-8)
sol1=sol[:,1]
sol2=np.log(np.abs(1-sol[:,1]))
x=(np.linspace(0,5,501))
x2=np.log(x)
plt.plot(x, sol1, label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig11.eps', format='eps')
plt.show()
plt.plot(x[300:], sol2[300:], label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$\log{(\mid 1-f}$'+" "+'$\'(\eta)\mid)$')
plt.savefig('fig12.eps', format='eps')
plt.show()


# In[595]:


sol=falkner_skan_integrator_type1(1,140/100,np.linspace(0,10,501),1e-8,1e-8)
sol1=sol[:,1]
sol2=np.log(np.abs(1-sol[:,1]))
x=(np.linspace(0,5,501))
x2=np.log(x)
plt.plot(x, sol1, label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+" "+'$\'(\eta)$')
plt.savefig('fig13.eps', format='eps')
plt.show()
plt.plot(x[300:], sol2[300:], label='f\'')
plt.xlabel('$\eta$')
plt.ylabel('$\log{(\mid 1-f}$'+" "+'$\'(\eta)\mid)$')
plt.savefig('fig14.eps', format='eps')
plt.show()


# ## (c) Plot and Tabulated Values of $S_m$

# In[169]:


m_vals=[]
Sm_vals=[]
for x_ in range(0,101,2):
    x=x_/100
    m_vals.append(x)
    Sm_vals.append(falkner_skan_shooting_method(x,1.4))
    print(m_vals[-1],Sm_vals[-1])
plt.plot(m_vals, Sm_vals , 'r', label='$S_m$')
plt.xlabel('$m$')
plt.ylabel('$S_m$')
plt.savefig('fig5.eps', format='eps')
plt.show()


# # Question 4

# # (a) Different possible terminal behaviours

# ### (a) (i) Algebraic Convergence

# In[596]:


sol=falkner_skan_integrator_type1(-0.1,0.5,np.linspace(0,40,501),1e-8,1e-8)
sol1=sol[:,1]
sol2=np.log(sol1-1)
x=np.linspace(0,40,501)
x2=np.log(x)
plt.plot(x,sol1, 'b')
plt.axhline(1,label='Asymptote, $y=1.0$',color='g')
plt.legend()
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig16.eps', format='eps')
plt.show()
plt.plot(x2[200:],sol2[200:], 'b')
plt.xlabel('$\log{(\eta)}$')
plt.ylabel('$\log{(\mid 1-f}$'+" "+'$\'(\eta)\mid)$')
plt.savefig('fig17.eps', format='eps')
plt.show()


# ### (a) (ii) Exponential Convergence

# In[597]:


ah=falkner_skan_shooting_method(-0.06,1)
sol=falkner_skan_integrator_type1(-0.06,ah,np.linspace(0,10,501),1e-8,1e-8)
sol1=sol[:,1]
sol2=np.log(np.abs(sol1-1))
x=np.linspace(0,10,501)
x2=np.log(x)
plt.plot(x,sol1, 'b')
plt.axhline(1,label='Asymptote, $y=1.0$', color='g')
plt.legend()
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig18.eps', format='eps')
plt.show()
plt.plot(x[300:],sol2[300:], 'b')
plt.xlabel('$\eta$')
plt.ylabel('$\log{(\mid 1-f}$'+" "+'$\'(\eta)\mid)$')
plt.savefig('fig19.eps', format='eps')
plt.show()


# ### (a) (iii) Finite Distance Singularity

# In[598]:


sol=falkner_skan_integrator_type1(-0.2,-0.5,np.linspace(0,6.435,501),1e-10,1e-10)
sol1=sol[:,1]
sol2=np.log(np.abs(sol1-1))
x=np.linspace(0,6.435,501)
x2=np.log(x)
plt.plot(x,sol1, 'b')
plt.legend()
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig20.eps', format='eps')
plt.show()


# ## (b) (i) First Branch of $S_m$

# In[182]:


m_vals_b1=[]
Sm_vals_b1=[]
for x_ in range(-90500,1,1):
    x=x_/1000000
    m_vals_b1.append(x)
    Sm_vals_b1.append(falkner_skan_shooting_method(x,0.5))
    #print(x,Sm_vals[-1])


# In[184]:


plt.plot(m_vals_b1[71:], Sm_vals_b1[71:],'r')
plt.xlabel('$m$')
plt.ylabel('$S_m$')
plt.savefig('fig21.eps', format='eps')
plt.show()


# In[186]:


for x in range(0,80):
    print(m_vals_b1[x],Sm_vals_b1[x])


# In[185]:


for x in range(71,len(m_vals)):
    print(m_vals_b1[x],Sm_vals_b1[x])


# ## (b) (ii) Second Branch of $S_m$

# In[366]:


m_vals_b2=[]
Sm_vals_b2=[]
for x_ in range(-910,1,15):
    x=x_/10000
    m_vals_b2.append(x)
    initval=-0.01
    if(x>-0.088 and x<-0.08):
        initval=-0.1
    if(x<-0.088 or x>-0.001):
        initval=-0.0001
    a1=falkner_skan_shooting_method2(x,initval)
    a2=falkner_skan_shooting_method2(x,initval)
    a3=falkner_skan_shooting_method2(x,initval)
    a=0
    if(a1==a2 or a1==a3):
        a=a1
    if(a2==a3 and a==0):
        a=a2
    Sm_vals_b2.append(a)


# In[563]:


m_vals_b2_2=[]
Sm_vals_b2_2=[]
for x_ in range(-10,-3,1):
    x=x_/10000
    m_vals_b2_2.append(x)
    initval=-0.005
    a1=falkner_skan_shooting_method2(x,initval)
    a2=falkner_skan_shooting_method2(x,initval)
    a3=falkner_skan_shooting_method2(x,initval)
    a=0
    if(a1==a2 or a1==a3):
        a=a1
    if(a2==a3 and a==0):
        a=a2
    Sm_vals_b2_2.append(a)
for x_ in range(-3,0,1):
    x=x_/10000
    m_vals_b2_2.append(x)
    initval=-0.0005
    a1=falkner_skan_shooting_method2(x,initval)
    a2=falkner_skan_shooting_method2(x,initval)
    a3=falkner_skan_shooting_method2(x,initval)
    a=0
    if(a1==a2 or a1==a3):
        a=a1
    if(a2==a3 and a==0):
        a=a2
    Sm_vals_b2_2.append(a)
for x_ in range(-9,2,2):
    x=x_/100000
    m_vals_b2_2.append(x)
    initval=-0.0005
    a1=falkner_skan_shooting_method2(x,initval)
    a2=falkner_skan_shooting_method2(x,initval)
    a3=falkner_skan_shooting_method2(x,initval)
    a=0
    if(a1==a2 or a1==a3):
        a=a1
    if(a2==a3 and a==0):
        a=a2
    Sm_vals_b2_2.append(a)


# In[564]:


plt.plot(m_vals_b2, Sm_vals_b2 , 'r', label='$S_m$')
plt.plot(m_vals_b2_2, Sm_vals_b2_2 , 'r')
plt.axhline(0,label='$y=0$',color='g')
plt.xlabel('$m$')
plt.ylabel('$S_m$')
plt.legend()
plt.savefig('fig22.eps', format='eps')
plt.show()


# In[566]:


m_vals_b2_2[-20:]


# In[567]:


Sm_vals_b2_2[-20:]


# ## (b) (iii) The plots of $f'(\eta)$ for different branches for $m=-0.05$

# In[653]:


m=-0.05
beta=2*m/(1+m)
alpha=np.sqrt((1+m)/2)
val=falkner_skan_shooting_method3(beta,0.5)
print(val*alpha)
sol=falkner_skan_integrator_type3(beta,val,np.linspace(0,20,501),1e-8,1e-8)
plt.plot(np.linspace(0,20,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig23.eps', format='eps')
plt.show()


# In[658]:


m=-0.05
beta=2*m/(1+m)
alpha=np.sqrt((1+m)/2)
val=falkner_skan_shooting_method3(beta,-0.1)
print(val*alpha)
sol=falkner_skan_integrator_type3(beta,val,np.linspace(0,20,501),1e-8,1e-8)
plt.plot(np.linspace(0,20,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig24.eps', format='eps')
plt.show()


# In[765]:


m=-0.0005
alpha=np.sqrt((1+m)/2)
val=falkner_skan_shooting_method2(m,-0.0005)
print(val)
sol=falkner_skan_integrator_type1(m,val,np.linspace(0,40,501),1e-8,1e-8)
plt.plot(np.linspace(0,40,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig25.eps', format='eps')
plt.show()


# In[768]:


m=0
val=falkner_skan_shooting_method(m,0.5)
print(val)
sol=falkner_skan_integrator_type1(m,val,np.linspace(0,20,501),1e-8,1e-8)
plt.plot(np.linspace(0,20,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig26.eps', format='eps')
plt.show()


# In[98]:


m=-0.2
beta=2*m/(1+m)
alpha=np.sqrt((1+m)/2)
val=falkner_skan_shooting_method3(beta,0.1)
print(val*alpha)
sol=falkner_skan_integrator_type3(beta,val,np.linspace(0,20,501),1e-8,1e-8)
plt.axhline(1,label='$y=0$',color='g')
plt.plot(np.linspace(0,20,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.legend()
plt.savefig('fig27.eps', format='eps')
plt.show()


# In[753]:


# val=falkner_skan_shooting_method2(-0.0001,-0.0005)
# print(val)
# val=val/alpha
# m=-0.0001
# beta=2*m/(1+m)
# alpha=np.sqrt((1+m)/2)
# sol=falkner_skan_integrator_type3(beta,val,np.linspace(0,100,501),1e-8,1e-8)
# plt.plot(np.linspace(0,100,501),sol[:,1])
# plt.xlabel('$\eta$')
# plt.ylabel('$f$'+' ' + '$\'(\eta)$')
# plt.show()


# ## (c) $-1<m<m_c$

# ### (c) (i) Another Branch

# In[83]:


m_vals_b4=[]
Sm_vals_b4=[]
for x_ in range(-1350,-1020,20):
    x=x_/1000
    m=x/(2-x)
    m_vals_b4.append(m)
    alpha=np.sqrt((1+m)/2)
    Sm_vals_b4.append(falkner_skan_shooting_method3(x,-1)*alpha)
    #print(m_vals[-1],Sm_vals[-1])
x_=-1000
x=x_/1000
m=x/(2-x)
m_vals_b4.append(m)
alpha=np.sqrt((1+m)/2)
Sm_vals_b4.append(falkner_skan_shooting_method3(x,-1.1)*alpha)


# In[118]:


for x in range(len(m_vals_b4)):
    print(m_vals_b4[x], Sm_vals_b4[x])


# In[84]:


plt.plot(m_vals_b4, Sm_vals_b4 , 'r')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.legend()
plt.savefig('fig28.eps', format='eps')
plt.show()


# In[152]:


m=m_vals_b4[-2]
beta=2*m/(1+m)
alpha=np.sqrt((1+m)/2)
val=Sm_vals_b4[-2]
print(m,val,alpha)
sol=falkner_skan_integrator_type1(m,val,np.linspace(0,20,501),1e-8,1e-8)
plt.axhline(1,label='$y=0$',color='g')
plt.plot(np.linspace(0,20,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.legend()
plt.savefig('fig30.eps', format='eps')
plt.show()


# In[132]:


m_vals_b4[1]


# In[166]:


m=m_vals_b4[2]
beta=2*m/(1+m)
alpha=np.sqrt((1+m)/2)
val=Sm_vals_b4[2]
print(m,val,alpha)
sol=falkner_skan_integrator_type1(m,val,np.linspace(0,20,501),1e-8,1e-8)
plt.axhline(1,label='$y=0$',color='g')
plt.plot(np.linspace(0,20,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.legend()
plt.savefig('fig31.eps', format='eps')
plt.show()


# ### (c) (ii) Another Another Branch

# In[107]:


m_vals_b3=[]
Sm_vals_b3=[]
for x_ in range(-2420,-2000,20):
    x=x_/1000
    m=x/(2-x)
    m_vals_b3.append(m)
    alpha=np.sqrt((1+m)/2)
    Sm_vals_b3.append(falkner_skan_shooting_method3(x,-1.2)*alpha)
    #print(m_vals[-1],Sm_vals[-1])
x_=-1980
x=x_/1000
m=x/(2-x)
m_vals_b3.append(m)
alpha=np.sqrt((1+m)/2)
Sm_vals_b3.append(falkner_skan_shooting_method3(x,-2)*alpha)


# In[108]:


plt.plot(m_vals_b3, Sm_vals_b3 , 'r')
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.legend()
plt.savefig('fig29.eps', format='eps')
plt.show()


# In[158]:


m_vals_b3[-1]


# In[171]:


m=m_vals_b3[-2]
beta=2*m/(1+m)
alpha=np.sqrt((1+m)/2)
val=Sm_vals_b4[-2]
print(m,val)
sol=falkner_skan_integrator_type1(m,val,np.linspace(0,15,501),1e-8,1e-8)
plt.axhline(1,label='$y=0$',color='g')
plt.plot(np.linspace(0,15,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.legend()
plt.savefig('fig32.eps', format='eps')
plt.show()


# In[173]:


m=m_vals_b3[1]
beta=2*m/(1+m)
alpha=np.sqrt((1+m)/2)
val=Sm_vals_b4[1]
print(m,val)
sol=falkner_skan_integrator_type1(m,val,np.linspace(0,10,501),1e-8,1e-8)
plt.axhline(1,label='$y=0$',color='g')
plt.plot(np.linspace(0,10,501),sol[:,1])
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.legend()
plt.savefig('fig33.eps', format='eps')
plt.show()


# # Question 5

# ## $sgn A=-1$

# In[229]:


falkner_skan_shooting_method_objective_m0(1.5,x,reltol=1e-8,abstol=1e-8)


# In[3]:


#---start of def---
def falkner_skan_setup_m0(f,nu):
    dfdnu=[0,0,0]
    dfdnu[0]=f[1]
    dfdnu[1]=f[2]
    dfdnu[2]=-(f[1]**2)+1
    return dfdnu
#---end of def---
#---start of def---
def falkner_skan_integrator_type_m0(S,nuspan,reltol,abstol):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup_m0, f_init, nuspan,rtol=reltol,atol=abstol)
    return sol
#---end of def---
#---start of def---
def falkner_skan_shooting_method_objective_m0(S,nuspan,reltol=1e-8,abstol=1e-8):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup_m0, f_init, nuspan,rtol=reltol,atol=abstol)
    df= sol[:,1]
    return df[-1]+1
#---end of def---
#---start of def---
def falkner_skan_shooting_method_m0(initval,x):
    Sm, = fsolve(falkner_skan_shooting_method_objective_m0, initval, args=(x,),xtol=1e-8)
    return Sm
# np.info(scipy.optimize.fsolve)
#---end of def---
#---start of def---  
def falkner_skan_shooting_method_plot_m0(initval):
    S_ideal=falkner_skan_shooting_method_m0(initval);
    print('S_ideal',S_ideal)
    sol=falkner_skan_integrator_type_m0(m,S_ideal,np.linspace(0,10,501),1e-8,1e-8);
    print(sol[:,1][-1]-1);
    plt.plot(np.linspace(0,10,501), sol[:, 1], 'b', label='theta(t)')
#---end of def---


# In[145]:


x=np.linspace(0,200,501)
x1=np.linspace(0,10,501)
Sval=falkner_skan_shooting_method_m0(1.15465,x)
print(Sval)
sol=falkner_skan_integrator_type_m0(Sval,x1,1e-8,1e-8)
sol1=sol[:,1]
plt.plot(x1,sol1)
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig34.eps', format='eps')
plt.show()


# In[146]:


x=np.linspace(0,15,501)
x1=np.linspace(0,15,501)
Sval=falkner_skan_shooting_method_m0(-1.15465,x)
print(Sval)
sol=falkner_skan_integrator_type_m0(Sval,x1,1e-8,1e-8)
sol1=sol[:,1]
plt.plot(x1,sol1)
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig35.eps', format='eps')
plt.show()


# In[147]:


x=np.linspace(0,130,501)
x1=np.linspace(0,20,501)
Sval=0.5
print(Sval)
sol=falkner_skan_integrator_type_m0(Sval,x1,1e-8,1e-8)
sol1=sol[:,1]
plt.plot(x1,sol1)
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig36.eps', format='eps')
plt.show()


# In[148]:


x1=np.linspace(0,8.52,501)
Sval=-1.155
sol=falkner_skan_integrator_type_m0(Sval,x1,1e-8,1e-8)
sol1=sol[:,1]
plt.plot(x1,sol1)
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig37.eps', format='eps')
plt.show()


# In[114]:


x1=np.linspace(0,11.75,501)
Sval=1.155
sol=falkner_skan_integrator_type_m0(Sval,x1,1e-8,1e-8)
sol1=sol[:,1]
plt.plot(x1,sol1)
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig38.eps', format='eps')
plt.show()


# ## $sgn A=1$

# In[124]:


#---start of def---
def falkner_skan_shooting_method_objective_m1(S,nuspan,reltol=1e-8,abstol=1e-8):
    f_init=[0,0,S]
    sol=odeint(falkner_skan_setup_m0, f_init, nuspan,rtol=reltol,atol=abstol)
    df= sol[:,1]
    return df[-1]-1
#---end of def---
#---start of def---
def falkner_skan_shooting_method_m1(initval,x):
    Sm, = fsolve(falkner_skan_shooting_method_objective_m1, initval, args=(x,),xtol=1e-8)
    return Sm
# np.info(scipy.optimize.fsolve)
#---end of def---


# In[157]:


x=np.linspace(0,5,501)
x1=np.linspace(0,5,501)
Sval=-0.4
sol=falkner_skan_integrator_type_m0(Sval,x1,1e-8,1e-8)
sol1=sol[:,1]
plt.plot(x1,sol1)
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig39.eps', format='eps')
plt.show()


# In[158]:


x=np.linspace(0,5,501)
x1=np.linspace(0,5,501)
Sval=0.1
sol=falkner_skan_integrator_type_m0(Sval,x1,1e-8,1e-8)
sol1=sol[:,1]
plt.plot(x1,sol1)
plt.xlabel('$\eta$')
plt.ylabel('$f$'+' ' + '$\'(\eta)$')
plt.savefig('fig40.eps', format='eps')
plt.show()


# In[ ]:




