import pandas as pd
import numpy as np
import scipy as scipy
import matplotlib as matplotlib
import matplotlib.pyplot as plt

#### load degrees of freedom table

geffhefftab = pd.read_csv("geff_heff.tab", sep="\s+")

# check it has been read in by printing first 5 lines
#print(geffhefftab.head())

#### get an array of temperature values and heff values

temperaturevals = np.array(geffhefftab['Temperature'])
heffvals = np.array(geffhefftab['heff'])

### sort in ascending temperature values 

ind = np.argsort(temperaturevals)
temperaturevals = temperaturevals[ind]
heffvals = heffvals[ind]
hefftab = np.column_stack([temperaturevals, heffvals])

#print(hefftab[10])

#print(temperaturevals.ndim)
#print(heffvals.ndim)


from scipy.interpolate import PchipInterpolator
heffint = PchipInterpolator(temperaturevals, heffvals)
geffint = heffint

plt.plot(temperaturevals, heffvals, '*')
plt.plot(temperaturevals, heffint(temperaturevals), 'red')
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
ax.set_xlim([10**-7, 10**10])
ax.set_ylim([2, 200])
#plt.show()
plt.xlabel("T [GeV]")
plt.ylabel("g_eff")
#plt.savefig("geff.jpg")
plt.clf()


###### define a function for the hubble rate ##########

mpl = 1.22*10**19


def hubble(T):
	return((8*np.pi**3*geffint(T)/90)**0.5*T**2/mpl)
	
#print(hubble(100))
	

###### define a function for the entropy density ##########

def sentropy(T):
	return(2*np.pi**2*heffint(T)/45*T**3)
	
#print(sentropy(100))


### define a function for the DM number density in equilibrium

gdm = 1

def ndmeq(z,mdm):
	return( gdm*mdm**3/(2*np.pi**2*z)*scipy.special.kn(2,z) )
	
#print(ndmeq(10,100))

### define a function for the DM number density normalized to entropy in equilibrium

def Ydmeq(z,mdm):
	return( ndmeq(z,mdm)/sentropy(mdm/z) )
	
#print(Ydmeq(10,100))

### define a function for the derivative of gstar with respect to temperature

from scipy.differentiate import derivative
gstar = geffint


def gstarprime(T):
	dT = T/1000
	res=derivative(gstar, T, initial_step=dT)	
	return(res.df)

#print(gstar(10))
#print(gstarprime(10))


#### define the derivative factor for the entropy factor in the Boltzmann equation

def derivfactor(T):
	if 0.005 < T and T < 1000:
		return( (1 + T/(3*gstar(T))*gstarprime(T)) )
	else:
		return(1)

#print(derivfactor(0.001),derivfactor(0.2),derivfactor(2000))

#### make a plot to check the deriv factor 

derivfactorvals = []

for temperature in temperaturevals:
	derivfactorvals.append(derivfactor(temperature))
	
derivfactorvals = np.array(derivfactorvals)	

from scipy.interpolate import CubicSpline
derivfactorint = CubicSpline(temperaturevals, derivfactorvals)
	
plt.plot(temperaturevals, derivfactorvals, '*')
plt.plot(temperaturevals, derivfactorint(temperaturevals), 'cyan')
plt.xscale('log')
ax = plt.gca()
ax.set_xlim([10**-7, 10**10])
ax.set_ylim([0.9, 2])
#plt.show()
plt.xlabel("T [GeV]")
plt.ylabel("derivfactor")
#plt.savefig("derivfactor.jpg")
plt.clf()

############### READ IN STEIGMAN DATA #################

steigmantab = pd.read_csv("steigman.csv")

# check it has been read in by printing first 5 lines
#print(steigmantab.head())

# plot steigman data ####

steigmassvals =  np.array(steigmantab['mass'])
steigsigmavals =  np.array(steigmantab['crosssection'])

plt.plot(steigmassvals, steigsigmavals, 'red')
plt.xscale('log')
plt.xlabel("mass [GeV]")
plt.ylabel("cross section [cm^3/s]")
#plt.savefig("steigman.jpg")
plt.clf()

#print(steigmantab['crosssection'])

############### NOW DEFINE BOLTZMANN EQUATIONS ETC. #################

from scipy.integrate import solve_ivp

def bmannDMext(mdm,sigman,n):
	zstart=10
	zend=50
	mdmfix=mdm
	
	def WDMeq(z):
		return(np.log(Ydmeq(z,mdm)))
		
	def wprime(t,y):
		return(sentropy(mdmfix/t)*sigman/(t**n)*derivfactorint(mdmfix/t)/(hubble(mdmfix/t)*t)*(np.e**(2*WDMeq(t)-y)-np.e**(y)))
		
		
	wstart=WDMeq(zstart)

		
	sol = solve_ivp(wprime, t_span=[zstart,zend], y0=[wstart], method='LSODA', rtol=1e-12, atol=1e-14)
	
#	print(sol.y[-1,-1])
#	print(np.e**(sol.y[-1,-1])*mdm)
	
	zstart2=zend
	zend2=10**5
	
	wstart2=sol.y[-1,-1]
	
	def wprime2(t,y):
		return(sentropy(mdmfix/t)*sigman/(t**n)*derivfactorint(mdmfix/t)/(hubble(mdmfix/t)*t)*(-np.e**(y)))
		
	sol2 = solve_ivp(wprime2, t_span=[zstart2,zend2], y0=[wstart2], method='LSODA', rtol=1e-12, atol=1e-14)
	
#	print(sol2.y[-1,-1])
#	print(np.e**(sol2.y[-1,-1])*mdm)
	
	return(np.e**(sol2.y[-1,-1])*mdm)

#
sigmadm = 1.86403*10**-9
#print(sigmadm)

#print(bmannDMext(1000,sigmadm,0))


##### find the required cross section #####

from scipy.optimize import elementwise

def sigmareqext(mdm,n):

	sigmamult = 1
	if -1/2-0.1 < n and  n < -1/2+0.1:
		sigmamult = 1/8
	elif n == 1:
		sigmamult = 40
	elif n == 2:
		sigmamult = 40**2
	else:
		sigmamult = 1
		
#	print(sigmamult)
	
	dmyieldtabx = []
	dmyieldtaby = []
	
	irange = np.linspace(-0.5,0.5,11)
	for i in irange:
		dmyieldtabx.append(10**float(i)*sigmamult*sigmadm)
		dmyieldtaby.append(bmannDMext(mdm,10**float(i)*sigmamult*sigmadm,n))
		
#	print(irange)
#	print(dmyieldtabx)
#	print(dmyieldtaby)
		
	dmyieldint = CubicSpline(dmyieldtabx, dmyieldtaby)
	
	def dmyieldminobs(sigma):
		return(dmyieldint(sigma) - 0.43e-9)
		
#	print(dmyieldminobs(sigmadm))
	
	res_root = elementwise.find_root(dmyieldminobs, (dmyieldtabx[0],dmyieldtabx[-1]))
	return(res_root.x)
	
#print(sigmareqext(1000,-1/2))
#print(sigmareqext(1000,0))
#print(sigmareqext(1000,1))
#print(sigmareqext(1000,2))

#################### Now add the perturbation effect #########################

def Psi(Ri,delta,zHbar,zbar):
	return( 2*np.abs(Ri)*np.cos(delta)*(np.sin(zbar/zHbar) - (zbar/zHbar)*np.cos(zbar/zHbar))/(zbar/zHbar)**3 )
	
#print(Psi(0.1,-0.1,0.4,10))

def deltarho(Ri,delta,zHbar,zbar):
	return(  8*np.abs(Ri)*np.cos(delta)/(zbar/zHbar)**3*(np.sin(zbar/zHbar) - (zbar/zHbar)*np.cos(zbar/zHbar) - (zbar/zHbar)**2*np.sin(zbar/zHbar) + 1/2*(zbar/zHbar)**3*np.cos(zbar/zHbar) ) )
	
#print(deltarho(0.1,-0.1,0.4,10))

def derivfactor2(T):
	if 0.005 < T and T < 1000:
		return( (1 + T/(4*gstar(T))*gstarprime(T)) )
	else:
		return(1)

derivfactorvals2 = []

for temperature in temperaturevals:
	derivfactorvals2.append(derivfactor2(temperature))
	
derivfactorvals2 = np.array(derivfactorvals2)	

from scipy.interpolate import CubicSpline
derivfactorint2 = CubicSpline(temperaturevals, derivfactorvals2)
		
#print(derivfactor2(0.1))

def deltaTgstar(Ri, delta, zHbar, zbar, T): 
 	return( deltarho(Ri, delta, zHbar, zbar)*1/(4*derivfactorint2(T)) )
 	
def deltaT(Ri, delta, zHbar, zbar): 
 	return( deltarho(Ri, delta, zHbar, zbar)*1/(4) )
 	
#print(deltaTgstar(0.1,-0.1,0.4,10,0.1))

#print(deltaT(0.1,-0.1,0.4,10))

def zT(zbar, deltaT): 
	return( zbar/(1 + deltaT) )

###### Boltzmann equation with the perturbation effect #####################

def bmannDMDText(mdm, sigman, n, Ri, delta, zHbar):
	zstart = 10
	zend = 50
	mdmfix = mdm
	
	def WDMeq(z):
  		return(np.log(Ydmeq(z,mdm)))
		
	def wprime(t,y):
		return( sentropy(mdmfix/zT(t, deltaTgstar(Ri, delta, zHbar, t, mdmfix/t)))*sigman/(zT(t, deltaTgstar(Ri, delta, zHbar, t, mdmfix/t))**n)*derivfactorint(mdmfix/zT(t, deltaTgstar(Ri, delta, zHbar, t, mdmfix/t)))/((1 - Psi(Ri, delta, zHbar, t))*hubble(mdmfix/t)*t)*(np.e**(2*WDMeq(zT(t, deltaTgstar(Ri, delta, zHbar, t, mdmfix/t)))-y)-np.e**(y)) )
			
	wstart=WDMeq(zstart)

	sol = solve_ivp(wprime, t_span=[zstart,zend], y0=[wstart], method='LSODA', rtol=1e-10, atol=1e-12, dense_output=True)
	
#	print(sol.y[-1,-1])
#	print(np.e**(sol.y[-1,-1])*mdm)
	
	zstart2=zend
	zend2=10**5
	
	wstart2=sol.y[-1,-1]
	
	def wprime2(t,y):
		return( sentropy(mdmfix/zT(t, deltaTgstar(Ri, delta, zHbar, t, mdmfix/t)))*sigman/(zT(t, deltaTgstar(Ri, delta, zHbar, t, mdmfix/t))**n)*derivfactorint(mdmfix/zT(t, deltaTgstar(Ri, delta, zHbar, t, mdmfix/t)))/((1 - Psi(Ri, delta, zHbar, t))*hubble(mdmfix/t)*t)*(-np.e**(y)) )
		
	sol2 = solve_ivp(wprime2, t_span=[zstart2,zend2], y0=[wstart2], method='LSODA', rtol=1e-10, atol=1e-12, dense_output=True)
	
#	print(sol2.y[-1,-1])
#	print(np.e**(sol2.y[-1,-1])*mdm)

	global solcomp
	def  solcomp(z):
		if z < zend:
			return(np.e**(sol.sol(z)))
		else:
			return(np.e**(sol2.sol(z)))
	
	return(np.e**(sol2.y[-1,-1])*mdm)	
	


###### Generate figure 2 #####################


sigmadm = 1.86403*10**-9	

def yobs(mdm):
	return(0.43e-9/mdm)

bmannDMDText(1000, sigmadm, 0, 0.2, 0, 0.25)
#print(solcomp(10**5))

x = np.linspace(10, 250, 2500)
y = []

for i in x:
	y.append(solcomp(i))
		
yeqnopert = Ydmeq(x,1000)

yeqpert = []
for i in x:
	yeqpert.append(Ydmeq(zT(i, deltaTgstar(0.2, 0, 0.25, i, 1000/i)),1000))


plt.title("$|R_i|=0.2$, $\\delta=0$, \n $\\sigma_0=1.9 \\times 10^{-9} \\mathrm{GeV}^{-2}$ $m_{\\mathrm{DM}} = 1$ TeV, $\\bar{x}_{H}=0.25$")
plt.plot(x, y, 'blue', label=r"$Y_X$")
plt.plot(x, np.full(x.shape, yobs(1000)), 'gray', label=r"$Y_{\mathrm{DM}} m_{\mathrm{DM}} = 0.43$ eV")
plt.plot(x, yeqnopert, 'orange', label=r"$Y_X^{\mathrm{eq}}(\delta_T = 0)$")
plt.plot(x, yeqpert, 'green', label=r"$Y_X^{\mathrm{eq}}$")
plt.legend(loc="upper right")
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
ax.set_xlim([10, 250])
ax.set_ylim([1e-15, 1e-5])
plt.xlabel(r"$\bar{x} = m_{\mathrm{DM}}/\bar{T}$")
plt.ylabel(r"$Y = n_{\mathrm{DM}}/s$")
plt.savefig("Fig2a.jpg")
plt.clf()


bmannDMDText(1000, sigmadm, 0, 0.2, 0, 1)
#print(solcomp(10**5))



x = np.linspace(10, 250, 2500)
y = []

for i in x:
	y.append(solcomp(i))
		
yeqnopert = Ydmeq(x,1000)

yeqpert = []
for i in x:
	yeqpert.append(Ydmeq(zT(i, deltaTgstar(0.2, 0, 1, i, 1000/i)),1000))
	
def yobs(mdm):
	return(0.43e-9/mdm)


plt.title("$|R_i|=0.2$, $\\delta=0$, \n $\\sigma_0=1.9 \\times 10^{-9} \\mathrm{GeV}^{-2}$ $m_{\\mathrm{DM}} = 1$ TeV, $\\bar{x}_{H}=1$")
plt.plot(x, y, 'blue', label=r"$Y_X$")
plt.plot(x, np.full(x.shape, yobs(1000)), 'gray', label=r"$Y_{\mathrm{DM}} m_{\mathrm{DM}} = 0.43$ eV")
plt.plot(x, yeqnopert, 'orange', label=r"$Y_X^{\mathrm{eq}}(\delta_T = 0)$")
plt.plot(x, yeqpert, 'green', label=r"$Y_X^{\mathrm{eq}}$")
plt.legend(loc="upper right")
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
ax.set_xlim([10, 250])
ax.set_ylim([1e-15, 1e-5])
plt.xlabel(r"$\bar{x} = m_{\mathrm{DM}}/\bar{T}$")
plt.ylabel(r"$Y = n_{\mathrm{DM}}/s$")
plt.savefig("Fig2b.jpg")
plt.clf()


bmannDMDText(1000, sigmadm, 0, 0.2, 0, 5)
#print(solcomp(10**5))



x = np.linspace(10, 250, 2500)
y = []

for i in x:
	y.append(solcomp(i))
		
yeqnopert = Ydmeq(x,1000)

yeqpert = []
for i in x:
	yeqpert.append(Ydmeq(zT(i, deltaTgstar(0.2, 0, 5, i, 1000/i)),1000))
	
def yobs(mdm):
	return(0.43e-9/mdm)

plt.title("$|R_i|=0.2$, $\\delta=0$, \n $\\sigma_0=1.9 \\times 10^{-9} \\mathrm{GeV}^{-2}$ $m_{\\mathrm{DM}} = 1$ TeV, $\\bar{x}_{H}=5$")
plt.plot(x, y, 'blue', label=r"$Y_X$")
plt.plot(x, np.full(x.shape, yobs(1000)), 'gray', label=r"$Y_{\mathrm{DM}} m_{\mathrm{DM}} = 0.43$ eV")
plt.plot(x, yeqnopert, 'orange', label=r"$Y_X^{\mathrm{eq}}(\delta_T = 0)$")
plt.plot(x, yeqpert, 'green', label=r"$Y_X^{\mathrm{eq}}$")
plt.legend(loc="upper right")
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
ax.set_xlim([10, 250])
ax.set_ylim([1e-15, 1e-5])
plt.xlabel(r"$\bar{x} = m_{\mathrm{DM}}/\bar{T}$")
plt.ylabel(r"$Y = n_{\mathrm{DM}}/s$")
plt.savefig("Fig2c.jpg")
plt.clf()

print("Finished making Figs. 2a, 2b, 2c.")


