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

import multiprocessing as mp
from multiprocessing import Pool

print("Number of cpus : ", mp.cpu_count())
pool = Pool(processes=(mp.cpu_count() - 1))

irange = np.linspace(-1,4,51)
cm3divs = 8.47287e16

##### scan over the mass values for s-wave
print("Starting s-wave calculation...")		

#def func1(i):
#	mdm = 10**i
#	return(sigmareqext(mdm,0)/cm3divs)

reqcrosssecx = 10**irange

args = [(10**i, 0) for i in irange]

with mp.Pool() as pool:
	reqcrosssectempy = pool.starmap(sigmareqext, args)
    
reqcrosssecy = []

for crosssec in reqcrosssectempy:
	reqcrosssecy.append(1/cm3divs*crosssec)


print("...s-wave done.")
##### scan over the mass values for d-wave
print("Starting Sommerfeld Enhanced calculation...")		

reqcrosssecSEx = 10**irange

args = [(10**i, -1/2) for i in irange]

with mp.Pool() as pool:
	reqcrosssectempSEy = pool.starmap(sigmareqext, args)
    
reqcrosssecSEy = []

for crosssec in reqcrosssectempSEy:
	reqcrosssecSEy.append(1/cm3divs*crosssec)

print("...Sommerfeld Enhanced done.")
##### scan over the mass values for d-wave
print("Starting p-wave calculation...")	

reqcrosssecPWAVEx = 10**irange

args = [(10**i, 1) for i in irange]

with mp.Pool() as pool:
	reqcrosssectempPWAVEy = pool.starmap(sigmareqext, args)
    
reqcrosssecPWAVEy = []

for crosssec in reqcrosssectempPWAVEy:
	reqcrosssecPWAVEy.append(1/cm3divs*crosssec)


print("...p-wave done.")
##### scan over the mass values for d-wave
print("Starting d-wave calculation...")	
	

reqcrosssecDWAVEx = 10**irange

args = [(10**i, 2) for i in irange]

with mp.Pool() as pool:
	reqcrosssectempDWAVEy = pool.starmap(sigmareqext, args)
    
reqcrosssecDWAVEy = []

for crosssec in reqcrosssectempDWAVEy:
	reqcrosssecDWAVEy.append(1/cm3divs*crosssec)

print("...d-wave done.")

#print(reqcrosssecSEy)
#print(reqcrosssecy)
#print(reqcrosssecPWAVEy)
#print(reqcrosssecDWAVEy)

plt.plot(reqcrosssecSEx, reqcrosssecSEy, 'cyan', label="SE")
plt.plot(reqcrosssecx, reqcrosssecy, 'blue', label="s-wave")
plt.plot(reqcrosssecPWAVEx, reqcrosssecPWAVEy, 'purple', label="p-wave")
plt.plot(reqcrosssecDWAVEx, reqcrosssecDWAVEy, 'red', label="d-wave")
plt.xscale('log')
plt.yscale('log')
plt.title("Standard Freeze-Out [No perturbations]")
plt.xlabel(r"$m_{\mathrm{DM}}$ [GeV]")
plt.ylabel(r"cross section $\sigma_n$ [cm$^3/$s]")
plt.legend(loc="upper right")
plt.savefig("Fig1_StandardFO.jpg")
plt.clf()

#plt.plot(reqcrosssecSEx, reqcrosssecSEy, 'blue')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel("DM mass [GeV]")
#plt.ylabel("cross section sigma_n [cm^3/s]")
#plt.savefig("StandardFO_SE.jpg")
#plt.clf()


print("Finished making Fig. 1")



