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


#################### Perturbation effects #########################

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

def Psideriv(Ri, delta, zHbar, zbar):
	return( 2*np.abs(Ri)*np.cos(delta)/(zbar/zHbar)**4*((zbar/zHbar)**2*np.sin(zbar/zHbar)-3*np.sin(zbar/zHbar) + 3*(zbar/zHbar)*np.cos(zbar/zHbar) ) )

def kV(Ri, delta, zHbar, zbar):
	return( -(3**0.5)*np.abs(Ri)*np.cos(delta)/(zbar/zHbar)**2*((zbar/zHbar)**2*np.sin(zbar/zHbar) - 2*np.sin(zbar/zHbar) + 2*(zbar/zHbar)*np.cos(zbar/zHbar)) )
	
############### NOW DEFINE BOLTZMANN EQUATIONS FOR THE ENTROPY #################

from scipy.integrate import solve_ivp

def entropysol(Ri, delta, zHbar):
	zstart = 10**-1
	zend = 10**2
	
	def yprime(t,y):
		return( y*(-3/t + 3**0.5/zHbar*kV(Ri, delta, zHbar, t) + 3/zHbar*Psideriv(Ri, delta, zHbar, t) ) )

	ystart = (1+3*deltaT(Ri, delta, zHbar, zstart) )
	
	sol = solve_ivp(yprime, t_span=[zstart,zend], y0=[ystart], method='LSODA', rtol=1e-13, atol=1e-13, dense_output=True)
	
	global solout
	def  solout(z):
		return(sol.sol(z)*(z/zstart)**3)
	
	return(sol.y[-1,-1])
	
############### MAKE FIGURE 6 #################

entropysol(0.2, 0, 1)

x = np.linspace(0.1, 50, 1000)
y = []
yanalytic1 = []
yanalytic2 = []

for i in x:
	y.append(solout(i))
	yanalytic1.append((1+3*deltaT(0.2,0,1,i)))
	yanalytic2.append((1+deltaT(0.2,0,1,i))**3)

	
ygray1 = []
ygray2 = []

for i in x:
	ygray1.append(0.4)
	ygray2.append(1.6)

plt.title("$|R_i|=0.2$, $\\delta=0$, $\\bar{x}_{H}=1$")
plt.plot(x, y, 'blue', label=r"$s/\bar{s}$ numerical")
plt.plot(x, yanalytic2, 'orange', linestyle='dashed', label=r"$(1+\delta_T)^3$ analytic")
plt.plot(x, yanalytic1, 'green', label=r"$(1+3\delta_T)$ analytic")
plt.plot(x, ygray1, 'gray')
plt.plot(x, ygray2, 'gray')
plt.legend(loc="upper left")
plt.xscale('log')
ax = plt.gca()
ax.set_xlim([0.1, 50])
ax.set_ylim([0, 2])
plt.xlabel(r"$\bar{x}$")
plt.ylabel(r"$\bar{s}/s$")
plt.savefig("Fig6.jpg")
plt.clf()

print("Finished making Fig. 6.")


