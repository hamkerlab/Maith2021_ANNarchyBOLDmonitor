#   Analysis of the computation time overhead created by the BOLD monitor.
#   The idea is to get an impression how much computational overhead this
#   recording introduce.
#
#   This is a part of the present manuskript:
#
#   Maith et al. (2021) BOLD monitoring in the neural simulator ANNarchy
#
#   author: Helge Uelo Dinkelbach
#
"""
General comments:

In advance you need to run a simulation with/without a BOLD recording, e. g.

python simulations.py 1 0 1 1 # without
python simulations.py 1 0 1 2 # with

In order to modify the number of neurons, you need to adjust the parameters.py:

params['corE_popsize'] = ... (line 88)
"""

#
# Experiment Setup
#

# this are the params['corE_popsize'] set in parameters.py
neuron_number = [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
# this is set by params['numInputs']
number_of_inputs = 10
# number of recorded neurons depends on the size of corE
labels = [ (x + 0.25*x) for x in neuron_number]

#
# Load the data from ../perfRaw (see the comments above how to create)
#
import numpy

w_bold_mean = numpy.zeros(len(neuron_number))
wo_bold_mean = numpy.zeros(len(neuron_number))

w_bold_std = numpy.zeros(len(neuron_number))
wo_bold_std = numpy.zeros(len(neuron_number))

for idx_n, n in enumerate(neuron_number):
    raw = numpy.recfromtxt("../perfRaw/with_monitor_"+str(n)+"_"+str(number_of_inputs)+"_1threads.csv")
    w_bold_mean[idx_n] = numpy.mean(raw)
    w_bold_std[idx_n] = numpy.std(raw)

    raw = numpy.recfromtxt("../perfRaw/without_monitor_"+str(n)+"_"+str(number_of_inputs)+"_1threads.csv")
    wo_bold_mean[idx_n] = numpy.mean(raw)
    wo_bold_std[idx_n] = numpy.std(raw)

#
# Plot computation as a function of the number of neurons
# and additionally the fraction as percent
#
from matplotlib.pylab import *
cm = 1/2.54  # centimeters to inches
rcParams['font.size'] = 8

fig, ax1 = plt.subplots(1,1,figsize=(17*cm,9*cm), dpi=300)
plt.subplots_adjust(top=0.97, bottom=0.2, left=0.1, right=0.92)
ax2 = ax1.twinx() 

ax1.plot(numpy.arange(len(neuron_number)), w_bold_mean, label="with BOLD recording")
print((w_bold_std/w_bold_mean) * 100)
ax1.plot(numpy.arange(len(neuron_number)), wo_bold_mean, label="without BOLD recording")
print((wo_bold_std/wo_bold_mean) * 100)

ax1.set_yscale("log")
# create a more readable y-axis
ax1.set_yticks([10**x for x in [0, 0.5, 1.0, 1.5, 2.0, 2.5]])
ax1.set_yticklabels([round(10**x,1) for x in [0, 0.5, 1.0, 1.5, 2.0, 2.5]])
ax1.set_ylabel("computation time [s, log-scale]", fontweight="bold")
ax1.set_xticks(np.arange(len(neuron_number)))
ax1.set_xticklabels([ str(round(x/1000.0,2)) for x in labels])
ax1.set_xlabel("number of recorded neurons [thousands]", fontweight="bold")
ax1.grid(True)
ax1.legend(ncol=2, bbox_to_anchor=(0.75,-0.15))

ax2.bar(numpy.arange(len(neuron_number)), ((w_bold_mean/wo_bold_mean)-1.0) * 100, width=0.2, color="gray", alpha=0.4)
ax2.set_ylabel("Fraction of computation time for\nBOLD recording [%]", color="gray", fontweight="bold")
ax2.tick_params(axis='y', colors='gray')

fig.savefig("../results/Fig_comp_time/FigS2.svg")
fig.savefig("../results/Fig_comp_time/FigS2.png")
