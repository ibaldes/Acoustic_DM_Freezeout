# Acoustic_DM_Freezeout
Code for the calculation of acoustically driven dark matter freeze-out. The corresponding paper can be found at arxiv.org/abs/2506.11884.
The code calculates the remaining dark matter at late times, after it has departed from thermal equilibrium due to the expandsion and cooling of the universe, and stopped annihilating. It finds the required cross section (controlling the annihilation rate) for the dark matter density to match the observed one. The key innovation of this project is that we included the possibility of large inhomogeneities in the early universe radiation (rather than a uniform density). This leads to sound waves in the plasma and effects the local temperature density of radiation and therefore the rate of dark matter annihilation. To find the required cross section in the presence of the oscillating radiation density requires solving the differential equation governing the dark matter freeze-out many more times than the standard freeze-out calculation (by about a factor of 15). This is because the initial amplitude of the oscillation varies from point to point in space (one can essentially think of a standing wave).

The different files generate the plots of 2506.11884 using python. (The plots in the arxiv version were orignally made using mathematica.) 
Uses pandas, numpy, scipy, multiprocessing, datetime and the geff_heff.tab file (for the effective relativistic degrees-of-freedom in the standard model bath).

Figs. 1, 2, and 6 are relatively fast to calculate.
Fig. 3 takes about 3 mins on my laptop.
Fig. 4 takes some 20-30 mins on my laptop.
Fig. 5 takes roughly 3.5 hours on my laptop (20 x Gen 12 Intel i7-12800H Cores).

