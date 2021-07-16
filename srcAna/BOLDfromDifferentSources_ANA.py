import numpy as np
import pylab as plt
import sys
from ANNarchy import raster_plot
from scipy import stats
from extras import set_size, get_pop_rate

font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 8}

plt.rc('font', **font)

def spikeActivityPlot(title='', spikes=None, simParams={}, times=[], ax=None):
    try:
        plt.title(title)
        # raster plot
        t,n = raster_plot(spikes)
        t*=simParams['dt']
        plt.plot(t,n,'k.',markersize=0.3)
        plt.xlim(times[0],times[-1])
        # population rate
        ax2=ax.twinx()
        firing_rate = get_pop_rate(spikes,simParams,simParams['sim_dur'],t_smooth_ms=-1)
        ax2.plot(times,firing_rate)
        return 1
    except:
        print('spikeActivityPlot',title,'did not work')
        return 0
        
def plotAverageOfNeuronVariables(title='', variables=[], labels=[], times=[], ax=None, simParams={}):
    if isinstance(variables, list):
        if len(variables)==1:
            plt.title(title)
            plt.plot(times, np.mean(variables[0],1), color='k')
            ## plot average of last 10 seconds
            if variables[0].shape[0]>int(10000/simParams['dt']):
                plt.axhline(np.mean(np.mean(variables[0],1)[-int(10000/simParams['dt']):]), color='r')
                plt.text((times[-1]+times[0])/2, np.mean(np.mean(variables[0],1)[-int(10000/simParams['dt']):]), str(round(np.mean(np.mean(variables[0],1)[-int(10000/simParams['dt']):]),3)), color='r', va='bottom')
            plt.xlim(times[0],times[-1])
            return 1
        elif len(variables)==2:
            # first averaged data
            plt.title(title)
            lns1 = plt.plot(times, np.mean(variables[0],1), color='C0', alpha=0.5, label=labels[0])
            plt.xlim(times[0],times[-1])
            # second averaged data
            ax2=ax.twinx()
            lns2 = ax2.plot(times, np.mean(variables[1],1), color='C1', alpha=0.5, label=labels[1])
            # legend
            lns = lns1+lns2
            ax.legend(lns, labels)
            return 1
        else:
            return 0
    else:
        return 0
        
def get_population_average_of_last_10(population_arr, simParams):
    """
        returns average value of last 10 seconds, given an population array, first idx = time, second index = neuron ranks
    """
    return np.mean(population_arr[-int(10000/simParams['dt']):,:])
    
def normalization_plot_column(title, mon_name, col, times, recordingsB_pulse, recordingsB_rest):
    """
        plots one column of the with vs without normalization plot
    """
    ### FIRST ROW
    ax=plt.subplot(3,2,col+1)
    plt.title(title)
    plt.plot(times, recordingsB_rest[mon_name+';I_CBF'][:,0]/np.max(recordingsB_pulse[mon_name+';I_CBF'][:,0]), color='black', label='resting')
    plt.plot(times, recordingsB_pulse[mon_name+';I_CBF'][:,0]/np.max(recordingsB_pulse[mon_name+';I_CBF'][:,0]), color='red', ls='dashed', label='pulse')
    plt.ylim(-0.3,1.05)
    ax.set_xticklabels([])
    if col==0: plt.ylabel('I')
    if col==1: ax.set_yticklabels([])
    ### SECOND ROW
    ax=plt.subplot(3,2,col+3)
    plt.plot(times, recordingsB_rest[mon_name+';f_in'][:,0], color='black')
    plt.plot(times, recordingsB_pulse[mon_name+';f_in'][:,0], color='red', ls='dashed')
    plt.ylim(0.9,1.35)
    ax.set_xticklabels([])
    if col==0: plt.ylabel('CBF')
    if col==1:
        ax.set_yticklabels([])
    ### THIRD ROW
    ax=plt.subplot(3,2,col+5)
    plt.plot(times, recordingsB_rest[mon_name+';BOLD'][:,0]*100, color='black', label='resting')
    plt.plot(times, recordingsB_pulse[mon_name+';BOLD'][:,0]*100, color='red', label='pulse', ls='dashed')
    plt.ylim(-0.5, 1.0)
    plt.xlabel('time [ms]')
    if col==0:
        plt.ylabel('BOLD [%]')
        plt.legend()
    if col==1:
        ax.set_yticklabels([])

def pulses_visualization_plot_row(row, ylabel, recordingsB, times, simParams):
    """
        plots one row of the pulses visualization plot
    """
    ### LEFT COLUMN
    plt.subplot(6,2,2*row+1)
    plt.ylabel(ylabel)
    plt.axvspan(simParams['rampUp']+simParams['sim_dur1'],simParams['rampUp']+simParams['sim_dur1']+simParams['sim_dur2'], color='k', alpha=0.3)
    if row<3:
        plt.plot(times, recordingsB[str(row+1)+';f_in'][:,0],label='CBF', color='k')
        if row==0: plt.title('CBF / CMRO2')
    else:
        plt.plot(times, recordingsB[str(row+1)+';f_in'][:,0],label='CBF', color='k')
        plt.plot(times, recordingsB[str(row+1)+';r'][:,0],label='CMRO2', color='grey', ls='dashed')
        if row==3: plt.legend()
        if row==5: plt.xlabel('time / ms')
    plt.ylim(0.8,1.75)
    ### RIGHT COLUMN
    plt.subplot(6,2,2*row+2)
    plt.axvspan(simParams['rampUp']+simParams['sim_dur1'],simParams['rampUp']+simParams['sim_dur1']+simParams['sim_dur2'], color='k', alpha=0.3)
    plt.plot(times, recordingsB[str(row+1)+';BOLD'][:,0],label='pulse', color='k')
    plt.ylim(-0.0165,0.0165)
    if row==0: plt.title('BOLD')
    if row==5: plt.xlabel('time / ms')

def get_firing_rates(spikes, dur):
    """
        spikes: list of ANNarchy spike dictionaries
        dur: length of time period in ms
        
        return firing rate distribution of population
    """
    
    ### collect all spike dicts into one spike dict
    spike_dict_all={}
    for idx, spike_dict in enumerate(spikes):
        for key, val in spike_dict.items():
            spike_dict_all[str(idx)+'_'+str(key)]=val
    
    ### obtain firing rate over all neurons of spike_dict_all
    rate = []
    for neuron, train in spike_dict_all.items():
        rate.append(len(train)/(dur/1000.))
    return np.array(rate)

def get_log_normal_fit(rates):
    """
        rates: array of firing rates
        
        return: fitted log normal distribution to firing rate distribution
    """
    
    ### fit lognorm to rates
    if len(rates[rates>=0])>0:
        shape,loc,scale = stats.lognorm.fit(rates[rates>=0])
        sigma=shape
        mu=np.log(scale)
    else:
        mu, sigma, loc = 0, 0.1, -2
    return np.array([mu,sigma,loc])

def lognormalPDF(x, mu=-0.702, sigma=0.9355, shift=0):
    """
        standard values from lognormal distribution of PSPs from Song et al. (2005)
    """
    x=x-shift
    if x.min()<=0:
        return np.concatenate([ np.zeros(x[x<=0].size) , np.exp(-(np.log(x[x>0]) - mu)**2 / (2 * sigma**2)) / (x[x>0] * sigma * np.sqrt(2 * np.pi)) ])
    else:
        return np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
    
        


def two_overview_plots(input_factor=1.0, stimulus=0, sim_id=''):
    ### LOAD DATA
    if len(sim_id)>0:
        load_string = str(input_factor).replace('.','_')+'_'+str(stimulus).replace('.','_')+'__'+sim_id
    else:
        load_string = str(input_factor).replace('.','_')+'_'+str(stimulus).replace('.','_')

    recordings  = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordings_'+load_string+'.npy', allow_pickle=True).item()
    recordingsB = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string+'.npy', allow_pickle=True).item()
    simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string+'.npy', allow_pickle=True).item()

    times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])



    ### CREATE FIGURE WITH STANDARD MONITORS
    plt.figure(figsize=(16,9),dpi=500)

    ## FIRST COLUMN
    ## SUBPLOT INPUT ACTIVITY
    ax = plt.subplot(2,5,1)
    spikeActivityPlot(title='input activity', spikes=recordings['inputPop;spike'], simParams=simParams, times=times, ax=ax)

    ## SECOND COLUMN
    ## SUBPLOT corE ACTIVITY
    ax = plt.subplot(2,5,2)
    spikeActivityPlot(title='corE activity', spikes=recordings['corEL1;spike'], simParams=simParams, times=times, ax=ax)
    ## SUBPLOT corI ACTIVITY
    ax = plt.subplot(2,5,7)
    spikeActivityPlot(title='corI activity', spikes=recordings['corIL1;spike'], simParams=simParams, times=times, ax=ax)

    ## THIRD COLUMN
    ## SUBPLOT corE RECORDED RATE
    ax = plt.subplot(2,5,3)
    plotAverageOfNeuronVariables(title='corE recorded rate', variables=[recordings['corEL1;r']], times=times, simParams=simParams)
    ## SUBPLOT corI RECORDED RATE
    ax = plt.subplot(2,5,8)
    plotAverageOfNeuronVariables(title='corI recorded rate', variables=[recordings['corIL1;r']], times=times, simParams=simParams)

    ## FOURTH COLUMN
    ## SUBPLOT corE SYN VARIABLE
    plt.subplot(2,5,4)
    plotAverageOfNeuronVariables(title='corE syn', variables=[recordings['corEL1;syn']], times=times, simParams=simParams)
    ## SUBPLOT corE SYN VARIABLE
    plt.subplot(2,5,9)
    plotAverageOfNeuronVariables(title='corI syn', variables=[recordings['corIL1;syn']], times=times, simParams=simParams)

    ## FIFTH COLUMN
    ## SUBPLOT corE var_r AND var_ra
    ax = plt.subplot(2,5,5)
    plotAverageOfNeuronVariables(title='corE var_r', variables=[recordings['corEL1;var_r'],recordings['corEL1;var_ra']], labels=['var_r','var_ra'], times=times, ax=ax, simParams=simParams)
    ## SUBPLOT corI var_r AND var_ra
    ax = plt.subplot(2,5,10)
    plotAverageOfNeuronVariables(title='corI var_r', variables=[recordings['corIL1;var_r'],recordings['corIL1;var_ra']], labels=['var_r','var_ra'], times=times, ax=ax, simParams=simParams)

    plt.tight_layout()
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_overview_standard.png')


    ### NEXT FIGURE WITH BOLD MONITORS
    plt.figure(figsize=(16,9),dpi=500)

    ## FIRST COLUMN
    ## BOLD1
    plt.subplot(5,6,1)
    plt.title('E raw input')
    plt.plot(times, recordingsB['1Eraw;I_CBF'][:,0])
    plt.subplot(5,6,7)
    plt.title('I raw input')
    plt.plot(times, recordingsB['1Iraw;I_CBF'][:,0])
    plt.subplot(5,6,13)
    plt.title('actual input'+str(round(np.mean(recordingsB['1;I_CBF'][:,0]),3)))
    plt.plot(times, recordingsB['1;I_CBF'][:,0])
    plt.subplot(5,6,19)
    plt.title('flow (solid), E (dashed)')
    plt.plot(times, recordingsB['1;f_in'][:,0],label='f_i')
    plt.plot(times, recordingsB['1;E'][:,0],label='E')
    plt.plot(times, recordingsB['1;q'][:,0],label='q')
    plt.plot(times, recordingsB['1;v'][:,0],label='v')
    plt.plot(times, recordingsB['1;f_out'][:,0],label='f_out')
    plt.legend()
    plt.subplot(5,6,25)
    plt.title('BOLD')
    plt.plot(times, recordingsB['1;BOLD'][:,0])

    ## SECOND COLUMN
    plt.subplot(5,6,2)
    plt.title('E raw input')
    plt.plot(times, recordingsB['2Eraw;I_CBF'][:,0])
    plt.subplot(5,6,8)
    plt.title('I raw input')
    plt.plot(times, recordingsB['2Iraw;I_CBF'][:,0])
    plt.subplot(5,6,14)
    plt.title('actual input'+str(round(np.mean(recordingsB['2;I_CBF'][:,0]),3)))
    plt.plot(times, recordingsB['2;I_CBF'][:,0])
    plt.subplot(5,6,20)
    plt.title('flow (solid), E (dashed)')
    plt.plot(times, recordingsB['2;f_in'][:,0],label='f_i')
    plt.plot(times, recordingsB['2;E'][:,0],label='E')
    plt.plot(times, recordingsB['2;q'][:,0],label='q')
    plt.plot(times, recordingsB['2;v'][:,0],label='v')
    plt.plot(times, recordingsB['2;f_out'][:,0],label='f_out')
    plt.legend()
    plt.subplot(5,6,26)
    plt.title('BOLD')
    plt.plot(times, recordingsB['2;BOLD'][:,0])

    ## THIRD COLUMN
    plt.subplot(5,6,3)
    plt.title('E raw input')
    plt.plot(times, recordingsB['3Eraw;I_CBF'][:,0])
    plt.subplot(5,6,9)
    plt.title('I raw input')
    plt.plot(times, recordingsB['3Iraw;I_CBF'][:,0])
    plt.subplot(5,6,15)
    plt.title('actual input'+str(round(np.mean(recordingsB['3;I_CBF'][:,0]),3)))
    plt.plot(times, recordingsB['3;I_CBF'][:,0])
    plt.subplot(5,6,21)
    plt.title('flow (solid), E (dashed)')
    plt.plot(times, recordingsB['3;f_in'][:,0],label='f_i')
    plt.plot(times, recordingsB['3;E'][:,0],label='E')
    plt.plot(times, recordingsB['3;q'][:,0],label='q')
    plt.plot(times, recordingsB['3;v'][:,0],label='v')
    plt.plot(times, recordingsB['3;f_out'][:,0],label='f_out')
    plt.legend()
    plt.subplot(5,6,27)
    plt.title('BOLD')
    plt.plot(times, recordingsB['3;BOLD'][:,0])

    ## FOURTH COLUMN
    plt.subplot(5,6,4)
    plt.title('E raw input')
    plt.plot(times, recordingsB['4Eraw;I_CBF'][:,0], label='I_f', alpha=0.5)
    plt.plot(times, recordingsB['4Eraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,10)
    plt.title('I raw input')
    plt.plot(times, recordingsB['4Iraw;I_CBF'][:,0], label='I_f', alpha=0.5)
    plt.plot(times, recordingsB['4Iraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,16)
    plt.title('actual input')
    plt.plot(times, recordingsB['4;I_CBF'][:,0], label='I_CBF', alpha=0.5)
    plt.plot(times, recordingsB['4;I_CMRO2'][:,0], label='I_CMRO2', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,22)
    plt.title('flow and oxygen')
    plt.plot(times, recordingsB['4;f_in'][:,0], label='CBF', alpha=0.5)
    plt.plot(times, recordingsB['4;r'][:,0], label='CMRO2', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,28)
    plt.title('BOLD')
    plt.plot(times, recordingsB['4;BOLD'][:,0])

    ## FITH COLUMN
    plt.subplot(5,6,5)
    plt.title('E raw input')
    plt.plot(times, recordingsB['5Eraw;I_CBF'][:,0], label='I_f', alpha=0.5)
    plt.plot(times, recordingsB['5Eraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,11)
    plt.title('I raw input')
    plt.plot(times, recordingsB['5Iraw;I_CBF'][:,0], label='I_f', alpha=0.5)
    plt.plot(times, recordingsB['5Iraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,17)
    plt.title('actual input')
    plt.plot(times, recordingsB['5;I_CBF'][:,0], label='I_CBF', alpha=0.5)
    plt.plot(times, recordingsB['5;I_CMRO2'][:,0], label='I_CMRO2', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,23)
    plt.title('flow and oxygen')
    plt.plot(times, recordingsB['5;f_in'][:,0], label='CBF', alpha=0.5)
    plt.plot(times, recordingsB['5;r'][:,0], label='CMRO2', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,29)
    plt.title('BOLD')
    plt.plot(times, recordingsB['5;BOLD'][:,0])

    ## SIXTH COLUMN
    plt.subplot(5,6,6)
    plt.title('E raw input')
    plt.plot(times, recordingsB['6Eraw;I_CBF'][:,0], label='I_f', alpha=0.5)
    plt.plot(times, recordingsB['6Eraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,12)
    plt.title('I raw input')
    plt.plot(times, recordingsB['6Iraw;I_CBF'][:,0], label='I_f', alpha=0.5)
    plt.plot(times, recordingsB['6Iraw;I_CMRO2'][:,0], label='I_r', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,18)
    plt.title('actual input')
    plt.plot(times, recordingsB['6;I_CBF'][:,0], label='I_CBF', alpha=0.5)
    plt.plot(times, recordingsB['6;I_CMRO2'][:,0], label='I_CMRO2', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,24)
    plt.title('flow and oxygen')
    plt.plot(times, recordingsB['6;f_in'][:,0], label='CBF', alpha=0.5)
    plt.plot(times, recordingsB['6;r'][:,0], label='CMRO2', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,30)
    plt.title('BOLD')
    plt.plot(times, recordingsB['6;BOLD'][:,0])

    plt.tight_layout()
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_overview_BOLD.png')
    

    
def different_input_strengths():
    """
        x-axis = input_factor, y-axis = mean firing E and I, mean CBF, CMRO2, BOLD, I_CBF and I_CMRO2, mean over last 10 seconds
    """
    
    input_factor_list=[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    stimulus=0
    
    y_values = np.empty((len(input_factor_list),7,6))
    y_values[:,:] = np.nan
    for input_factor_idx,input_factor in enumerate(input_factor_list):
        ### LOAD DATA
        load_string = str(input_factor).replace('.','_')+'_'+str(stimulus).replace('.','_')
        recordings  = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordings_'+load_string+'.npy', allow_pickle=True).item()
        recordingsB = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string+'.npy', allow_pickle=True).item()
        simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string+'.npy', allow_pickle=True).item()

        times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])
        
        
        ### MEAN FIRING RATE
        y_values[input_factor_idx,0,:] = get_population_average_of_last_10(recordings['corEL1;r'],simParams)
        y_values[input_factor_idx,1,:] = get_population_average_of_last_10(recordings['corIL1;r'],simParams)
        

        ### MEAN CBF OF ALL BOLD MONITORS
        for bold_monitor in [1,2,3,4,5,6]:
            ## f_in of BOLD monitors
            y_values[input_factor_idx,2,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';f_in'],simParams)


        ### MEAN CMRO2 OF LAST BOLD MONITORS
        for bold_monitor in [4,5,6]:
            y_values[input_factor_idx,3,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';r'],simParams)

        
        ### MEAN BOLD OF ALL BOLD MONITORS
        for bold_monitor in [1,2,3,4,5,6]:
            y_values[input_factor_idx,4,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';BOLD'],simParams)


        ### MEAN I_CBF OF ALL BOLD MONITORS
        for bold_monitor in [1,2,3,4,5,6]:
            ## I_CBF of BOLD monitors
            y_values[input_factor_idx,5,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';I_CBF'],simParams)


        ### MEAN I_CMRO2 OF LAST BOLD MONITORS
        for bold_monitor in [4,5,6]:
            y_values[input_factor_idx,6,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';I_CMRO2'],simParams)


    ### PLOT
    num_rows=6
    num_cols=3
    plt.figure(figsize=(10,12), dpi=500)

    for bold_monitor in [1,2,3,4,5,6]:
        row=bold_monitor-1
        if row==0:
            ## FIRST ROW: FIRING RATES + BOLD MONITOR 1 + TITLES
            # FIRING RATES
            plt.subplot(num_rows,num_cols,row*num_cols+1)
            plt.title('firing rates')
            plt.plot(input_factor_list,y_values[:,0,row],color='k')
            plt.plot(input_factor_list,y_values[:,1,row],color='k',ls='dashed')
            plt.xlabel('input factor')
            # CBF AND CMRO2
            plt.subplot(num_rows,num_cols,row*num_cols+2)
            plt.title('CBF and CMRO2')
            plt.plot(input_factor_list,y_values[:,2,row],color='k')
            # BOLD
            plt.subplot(num_rows,num_cols,row*num_cols+3)
            plt.title('BOLD')
            plt.plot(input_factor_list,y_values[:,4,row],color='k')
        elif row==(num_rows-1):
            ## LAST ROW: LAST BOLD MONITOR + XLABELS
            # CBF AND CMRO2
            plt.subplot(num_rows,num_cols,row*num_cols+2)
            plt.plot(input_factor_list,y_values[:,2,row],color='k')
            plt.plot(input_factor_list,y_values[:,3,row],color='k',ls='dashed')
            plt.xlabel('input factor')
            # BOLD
            plt.subplot(num_rows,num_cols,row*num_cols+3)
            plt.plot(input_factor_list,y_values[:,4,row],color='k')
            plt.xlabel('input factor')
        else:
            ## OTHER ROWS: OTHER BOLD MONITORS
            # CBF AND CMRO2
            plt.subplot(num_rows,num_cols,row*num_cols+2)
            plt.plot(input_factor_list,y_values[:,2,row],color='k')
            if bold_monitor in [4,5,6]:
                plt.plot(input_factor_list,y_values[:,3,row],color='k',ls='dashed')
            # BOLD
            plt.subplot(num_rows,num_cols,row*num_cols+3)
            plt.plot(input_factor_list,y_values[:,4,row],color='k')

    plt.tight_layout()
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_different_input_strengths.png')



def with_vs_without_normalization(num_sims=1):
    """
        load data of one pulse simulation
        demonstrate input+flow+BOLD for standard BOLD monitor with and without normalization
        additionally show same data without a pulse stimulus
    """
    ### LOAD DATA
    load_string_pulse = '5_0_1'
    load_string_rest = '1_0_1'
    
    ## LOAD recordingsB num_sims TIMES AND AVERAGE THE RECORDINGS
    recordingsB_pulse={}
    recordingsB_rest={}
    for sim_id in range(num_sims):
        recordingsB_pulse_loaded = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string_pulse+'__'+str(int(sim_id))+'.npy', allow_pickle=True).item()
        recordingsB_rest_loaded = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string_rest+'__'+str(int(sim_id))+'.npy', allow_pickle=True).item()
        for key,val in recordingsB_pulse_loaded.items():
            try:
                recordingsB_pulse[key]+=recordingsB_pulse_loaded[key]/num_sims
            except:
                recordingsB_pulse[key]=recordingsB_pulse_loaded[key]/num_sims
        for key,val in recordingsB_rest_loaded.items():
            try:
                recordingsB_rest[key]+=recordingsB_rest_loaded[key]/num_sims
            except:
                recordingsB_rest[key]=recordingsB_rest_loaded[key]/num_sims
    
    simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string_pulse+'.npy', allow_pickle=True).item()

    times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])

    ### PLOT
    plt.figure(figsize=(8.5/2.54,8.5/2.54),dpi=500)

    ## WITHOUT NORMALIZATION
    normalization_plot_column(title='without normalization', mon_name='1withoutNorm', col=0, times=times, recordingsB_pulse=recordingsB_pulse, recordingsB_rest=recordingsB_rest)
    ## WITH NORMALIZATION
    normalization_plot_column(title='with normalization', mon_name='1', col=1, times=times, recordingsB_pulse=recordingsB_pulse, recordingsB_rest=recordingsB_rest)
    
    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.2)
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_with_vs_without_norm.svg')



def pulses_visualization(num_sims=1, load_string = '5_0_1'):
    """
        load data of one pulse simulation
        visualize CBF/CMRO2 and BOLD of the different BOLD monitors with different source signals
    """

    ### LOAD DATA
    
    ## LOAD recordingsB num_sims TIMES AND AVERAGE THE RECORDINGS
    recordingsB={}
    for sim_id in range(num_sims):
        recordingsB_loaded = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string+'__'+str(int(sim_id))+'.npy', allow_pickle=True).item()
        for key,val in recordingsB_loaded.items():
            try:
                recordingsB[key]+=recordingsB_loaded[key]/num_sims
            except:
                recordingsB[key]=recordingsB_loaded[key]/num_sims
    simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string+'__0.npy', allow_pickle=True).item()

    times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])


    ### PLOT WITH ALL DATA
    plt.figure(figsize=(10,12),dpi=500)

    for row, label in enumerate(['A','B','C','D','E','F']):
        pulses_visualization_plot_row(row, label, recordingsB, times, simParams)

    plt.tight_layout()
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_pulses_visualization.png', dpi=500)
    
    
    ### SINGLE PLOTS FOR MANUSCRIPT
    for row, label in enumerate(['A','B','C','D','E','F']):
    
        ### CBF/CMRO2 PLOTS
        plt.figure()
        plt.axvspan(simParams['rampUp']+simParams['sim_dur1'],simParams['rampUp']+simParams['sim_dur1']+simParams['sim_dur2'], color='k', alpha=0.3)
        if row<3:
            plt.plot(times, recordingsB[str(row+1)+';f_in'][:,0],label='CBF', color='red')
        else:
            plt.plot(times, recordingsB[str(row+1)+';f_in'][:,0],label='CBF', color='red')
            plt.plot(times, recordingsB[str(row+1)+';r'][:,0],label='CMRO2', color='blue', ls='dashed')
            if row==3: plt.legend()
        plt.ylim(0.8,1.75)
        plt.tight_layout(pad=10)
        set_size(4.89/2.54,2.08/2.54)
        plt.savefig('../results/BOLDfromDifferentSources/pulses_visu/CBF_CMRO2_'+label+'.svg')
        
        ### BOLD PLOTS
        plt.figure()
        plt.axvspan(simParams['rampUp']+simParams['sim_dur1'],simParams['rampUp']+simParams['sim_dur1']+simParams['sim_dur2'], color='k', alpha=0.3)
        plt.plot(times, recordingsB[str(row+1)+';BOLD'][:,0],label='BOLD', color='k')
        plt.ylim(-0.015,0.0165)
        plt.tight_layout(pad=10)
        set_size(4.89/2.54,2.08/2.54)
        plt.savefig('../results/BOLDfromDifferentSources/pulses_visu/BOLD_'+label+'.svg')



def BOLD_correlations():
    """
        load data of one long resting simulation
        load monitored BOLD signals of diffeent BOLD monitors
        generate correlation matrix between these BOLD signals
    """

    ### LOAD DATA
    load_string = '1_0_2'
    recordingsB = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string+'.npy', allow_pickle=True).item()
    simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string+'.npy', allow_pickle=True).item()

    times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])

    ### COMPUTE CORRELATIONS
    corr_mat = np.corrcoef([recordingsB[str(i+1)+';BOLD'][:,0] for i in range(6)])
    for i in range(6):
        for j in range(6):
            if i>j: corr_mat[i,j]=0

    ### FIRST PLOT ALL SIX BOLD SIGNALS
    plt.figure(figsize=(16,9), dpi=500)
    name=['A','B','C','D','E','F']
    for i in range(6):
        plt.subplot(6,1,i+1)
        plt.plot(times,recordingsB[str(i+1)+';BOLD'])
        plt.ylim(-0.01,0.01)
        plt.ylabel('BOLD '+name[i])
    plt.xlabel('time / ms')
    
    plt.tight_layout()
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_correlations_signals.png')

    ### SECOND PLOT CORRELATION MATRIX
    plt.figure(figsize=(10,10), dpi=500)
    ax = plt.subplot(111)
    plt.imshow(corr_mat, vmin=-1, vmax=1, cmap='bwr')
    plt.xticks(list(range(6)), ['A','B','C','D','E','F'])
    plt.yticks(list(range(6)), ['A','B','C','D','E','F'])
    ax.xaxis.set_ticks_position('top')
    for i in range(6):
        for j in range(6):
            if i<=j: plt.text(j,i,str(round(corr_mat[i,j],2)), ha='center', va='center')
    #plt.colorbar()
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_correlations_matrix.png')



def rate_distribution():
    """
        load data of standard simulation (no stimulus change)
        plot firing rate distribution of combined corEL1 and corIL1
    """

    ### LOAD DATA
    load_string = '1_0_0'
    recordings  = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordings_'+load_string+'.npy', allow_pickle=True).item()
    simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string+'.npy', allow_pickle=True).item()

    times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])
    
    ### OBTAIN FIRING RATES OF ALL NEURONS
    firing_rates = get_firing_rates([recordings['corEL1;spike'],recordings['corIL1;spike']], simParams['sim_dur'])
    
    ### OBTAIN FITTED LOG NORMAL DISTRIBUTION
    fit = get_log_normal_fit(firing_rates)
    
    ### PLOT HISTOGRAM AND FITTED AND EXPERIMENTAL DISTRIBUTION
    plt.figure()
    x=np.linspace(0,40,1000)

    plt.hist(firing_rates, 75, density=True, align='mid', alpha=0.5, color='grey')
    plt.plot(x,lognormalPDF(x, mu=1.2, sigma=1.1), label='Buzsaki & Mizuseki (2014)', color='k')
    plt.plot(x, lognormalPDF(x, mu=fit[0], sigma=fit[1], shift=fit[2]), label='Model', color='red', ls='dashed')
    plt.legend(fontsize=8)
    set_size(5.93/2.54,2.44/2.54)
    plt.savefig('../results/BOLDfromDifferentSources/BOLDfromDifferentSources_ANA_rate_distribution.svg')
    

    


if __name__=='__main__':

    overview_plot=0
    different_input_strengths_plot=0
    with_vs_without_norm_plot=0
    pulses_visu_plot=1
    correlation_plot=0
    rate_dist_plot=0
    
    if overview_plot:
        if len(sys.argv)==4:
            ## optional input_factor, stimulus and sim_id given
            two_overview_plots(input_factor=float(sys.argv[1]), stimulus=int(sys.argv[2]), sim_id=str(int(sys.argv[3])))
        elif len(sys.argv)==3:
            ## optional input_factor and stimulus given
            two_overview_plots(input_factor=float(sys.argv[1]), stimulus=int(sys.argv[2]))
        elif len(sys.argv)==2:
            ## optional input_factor given
            two_overview_plots(input_factor=float(sys.argv[1]))
        else:
            ## standard simulation arguments used
            two_overview_plots()
            
    if different_input_strengths_plot:
        different_input_strengths()

    if with_vs_without_norm_plot:
        with_vs_without_normalization(num_sims=20)

    if pulses_visu_plot:
        pulses_visualization(num_sims=20)#, load_string = '1_2_3'

    if correlation_plot:
        BOLD_correlations()
        
    if rate_dist_plot:
        rate_distribution()



















