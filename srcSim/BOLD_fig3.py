#   BOLD monitoring example to create Figure 3 of the present
#   manuskript:
#
#   Maith et al. (2021) BOLD monitoring in the neural simulator ANNarchy
#
#   author: Helge Uelo Dinkelbach, Oliver Maith
#
from ANNarchy import *
setup(seed=56789)       # fixed seed to allow easy fine-tune of axis-ticks etc.

from ANNarchy.extensions.bold import *
from matplotlib.pylab import *
from matplotlib.image import imread

rcParams['font.size'] = 8

# A population of 100 izhikevich neurons
pop0 = Population(100, neuron=Izhikevich)
pop1 = Population(100, neuron=Izhikevich)

# Set noise to create some baseline activity
pop0.noise = 5.0; pop1.noise = 5.0

# Compute mean firing rate in Hz on 100ms window
pop0.compute_firing_rate(window=100.0)
pop1.compute_firing_rate(window=100.0)

# Create required monitors
mon_pop0 = Monitor(pop0, ["r"], start=False)
mon_pop1 = Monitor(pop1, ["r"], start=False)
m_bold = BoldMonitor(
    populations=[pop0, pop1],               # recorded populations
    source_variables="r",                   # mean firing rate as input
    input_variables="exc",
    normalize_input=[2000, 2000],           # time window to compute baseline
                                            # should be multiple of fr-window
    recorded_variables=["sum(exc)", "BOLD"] # we want to analyze the BOLD input
)

# Compile and initialize the network
compile()

# Reach stable activity
simulate(1000)

# Start recording
mon_pop0.start()
mon_pop1.start()
m_bold.start()

# we manipulate the noise for the half of the neurons
simulate(5000)      # 5s with low noise
pop0.noise = 7.5
simulate(5000)      # 5s with higher noise (one population)
pop0.noise = 5
simulate(10000)     # 10s with low noise

# An example evaluation, which consists of:
# A) the mean firing activity for both populations
# B) the accumulated activity which serves as input to BOLD
# C) the resulting BOLD signal

# some figure stuff
cm = 1/2.54  # centimeters to inches
figure(figsize=(20*cm,9*cm), dpi=300)
grid = plt.GridSpec(1, 3, wspace=0.45, left=0.075, right=0.97)

# A) mean firing rate
ax1 = subplot(grid[0, 0])
mean_fr1 = np.mean(mon_pop0.get("r"), axis=1)
mean_fr2 = np.mean(mon_pop1.get("r"), axis=1)

ax1.plot(mean_fr1, label="pop0")
ax1.plot(mean_fr2, label="pop1")
ax1.text(-6000, 15.25, "A", fontweight="bold", fontsize=16)
legend()
ax1.set_ylabel("average mean firing rate [Hz]", fontweight="bold")

ymin = 1.5
ymax = 14.5
ax1.set_ylim([ymin,ymax])
ax1.vlines(2000, ymin, ymax, color='gray', linestyle='--')
ax1.vlines(5000, ymin, ymax, color='gray', linestyle='--')
ax1.vlines(10000, ymin, ymax, color='gray', linestyle='--')

# b) BOLD input signal
ax2 = subplot(grid[0, 1])

bold_data = m_bold.get("sum(exc)")
ax2.plot(bold_data, color="k")
ax2.set_ylabel("BOLD input 'exc'", fontweight="bold")
ax2.text(-8000, 0.97, "B", fontweight="bold", fontsize=16)
ax2.set_xticks(np.arange(0,21,2)*1000)
ax2.set_xticklabels(np.arange(0,21,2))

ymin = -0.3
ymax = 0.9
ax2.set_ylim([ymin,ymax])
ax2.vlines(2000, ymin, ymax, color='gray', linestyle='--')
ax2.vlines(5000, ymin, ymax, color='gray', linestyle='--')
ax2.vlines(10000, ymin, ymax, color='gray', linestyle='--')


# C) BOLD input signal in percent
ax3 = subplot(grid[0, 2])

bold_data = m_bold.get("BOLD")
ax3.plot(bold_data*100.0, color="k")
ax3.set_ylabel("BOLD [%]", fontweight="bold")
ax3.text(-10000, 3.2, "C", fontweight="bold", fontsize=16)

ymin = -1
ymax = 3
ax3.set_ylim([ymin,ymax])
ax3.set_ylim([-1, 3])
ax3.vlines(2000, ymin, ymax, color='gray', linestyle='--')
ax3.vlines(5000, ymin, ymax, color='gray', linestyle='--')
ax3.vlines(10000, ymin, ymax, color='gray', linestyle='--')


for ax in [ax1, ax2, ax3]:
    ax.set_xticks(np.arange(0,21,2)*1000)
    ax.set_xticklabels(np.arange(0,21,2))
    ax.set_xlabel("time [s]", fontweight="bold")

savefig("../results/Fig_simple_example/Fig3.svg")
savefig("../results/Fig_simple_example/Fig3.png")
