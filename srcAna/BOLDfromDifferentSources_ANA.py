import numpy as np
import pylab as plt
import sys
from ANNarchy import raster_plot
from initialTestofBOLD_ANA import get_pop_rate

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
        ax2.plot(times,get_pop_rate(spikes,simParams,simParams['sim_dur'],t_smooth_ms=-1))
        return 1
    except:
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
    plt.subplot(3,2,col+1)
    plt.title(title)
    plt.plot(times, recordingsB_pulse[mon_name+';r'][:,0]/np.max(recordingsB_pulse[mon_name+';r'][:,0]), color='k', label='pulse')
    plt.plot(times, recordingsB_rest[mon_name+';r'][:,0]/np.max(recordingsB_pulse[mon_name+';r'][:,0]), color='grey', ls='dashed', label='resting')
    plt.ylim(-0.3,1.05)
    if col==0: plt.ylabel('I')
    if col==1: plt.legend()
    ### SECOND ROW
    plt.subplot(3,2,col+3)
    plt.plot(times, recordingsB_pulse[mon_name+';f_in'][:,0], color='k')
    plt.plot(times, recordingsB_rest[mon_name+';f_in'][:,0], color='grey', ls='dashed')
    plt.ylim(0.87,1.27)
    if col==0: plt.ylabel('CBF')
    ### THIRD ROW
    plt.subplot(3,2,col+5)
    plt.plot(times, recordingsB_pulse[mon_name+';BOLD'][:,0], color='k')
    plt.plot(times, recordingsB_rest[mon_name+';BOLD'][:,0], color='grey', ls='dashed')
    plt.ylim(-0.01, 0.01)
    plt.xlabel('time / ms')
    if col==0: plt.ylabel('BOLD')

def pulses_visualization_plot_row(row, ylabel,recordingsB, times, simParams):
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
        plt.plot(times, recordingsB[str(row+1)+';CBF'][:,0],label='CBF', color='k')
        plt.plot(times, recordingsB[str(row+1)+';CMRO2'][:,0],label='CMRO2', color='grey', ls='dashed')
        if row==3: plt.legend()
        if row==5: plt.xlabel('time / ms')
    plt.ylim(0.85,1.7)
    ### RIGHT COLUMN
    plt.subplot(6,2,2*row+2)
    plt.axvspan(simParams['rampUp']+simParams['sim_dur1'],simParams['rampUp']+simParams['sim_dur1']+simParams['sim_dur2'], color='k', alpha=0.3)
    plt.plot(times, recordingsB[str(row+1)+';BOLD'][:,0],label='BOLD', color='k')
    plt.ylim(-0.005,0.015)
    if row==0: plt.title('BOLD')
    if row==5: plt.xlabel('time / ms')
        


def two_overview_plots(input_factor=1.0, stimulus=0):
    ### LOAD DATA
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
    plt.savefig('BOLDfromDifferentSources_ANA_overview_standard.png')


    ### NEXT FIGURE WITH BOLD MONITORS
    plt.figure(figsize=(16,9),dpi=500)

    ## FIRST COLUMN
    ## BOLD1
    plt.subplot(5,6,1)
    plt.title('E raw input')
    plt.plot(times, recordingsB['1Eraw;r'][:,0])
    plt.subplot(5,6,7)
    plt.title('I raw input')
    plt.plot(times, recordingsB['1Iraw;r'][:,0])
    plt.subplot(5,6,13)
    plt.title('actual input'+str(round(np.mean(recordingsB['1;r'][:,0]),3)))
    plt.plot(times, recordingsB['1;r'][:,0])
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
    plt.plot(times, recordingsB['2Eraw;r'][:,0])
    plt.subplot(5,6,8)
    plt.title('I raw input')
    plt.plot(times, recordingsB['2Iraw;r'][:,0])
    plt.subplot(5,6,14)
    plt.title('actual input'+str(round(np.mean(recordingsB['2;r'][:,0]),3)))
    plt.plot(times, recordingsB['2;r'][:,0])
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
    plt.plot(times, recordingsB['3Eraw;r'][:,0])
    plt.subplot(5,6,9)
    plt.title('I raw input')
    plt.plot(times, recordingsB['3Iraw;r'][:,0])
    plt.subplot(5,6,15)
    plt.title('actual input'+str(round(np.mean(recordingsB['3;r'][:,0]),3)))
    plt.plot(times, recordingsB['3;r'][:,0])
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
    plt.plot(times, recordingsB['4;CBF'][:,0], label='CBF', alpha=0.5)
    plt.plot(times, recordingsB['4;CMRO2'][:,0], label='CMRO2', alpha=0.5)
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
    plt.plot(times, recordingsB['5;CBF'][:,0], label='CBF', alpha=0.5)
    plt.plot(times, recordingsB['5;CMRO2'][:,0], label='CMRO2', alpha=0.5)
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
    plt.plot(times, recordingsB['6;CBF'][:,0], label='CBF', alpha=0.5)
    plt.plot(times, recordingsB['6;CMRO2'][:,0], label='CMRO2', alpha=0.5)
    plt.legend()
    plt.subplot(5,6,30)
    plt.title('BOLD')
    plt.plot(times, recordingsB['6;BOLD'][:,0])

    plt.tight_layout()
    plt.savefig('BOLDfromDifferentSources_ANA_overview_BOLD.png')
    

    
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
            ## f_in of first BOLD monitors and CBF of last monitors
            try:
                y_values[input_factor_idx,2,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';f_in'],simParams)
            except:
                y_values[input_factor_idx,2,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';CBF'],simParams)


        ### MEAN CMRO2 OF LAST BOLD MONITORS
        for bold_monitor in [4,5,6]:
            y_values[input_factor_idx,3,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';CMRO2'],simParams)

        
        ### MEAN BOLD OF ALL BOLD MONITORS
        for bold_monitor in [1,2,3,4,5,6]:
            y_values[input_factor_idx,4,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';BOLD'],simParams)


        ### MEAN I_CBF OF ALL BOLD MONITORS
        for bold_monitor in [1,2,3,4,5,6]:
            ## f_in of first BOLD monitors and CBF of last monitors
            try:
                y_values[input_factor_idx,5,bold_monitor-1] = get_population_average_of_last_10(recordingsB[str(bold_monitor)+';r'],simParams)
            except:
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
    plt.savefig('BOLDfromDifferentSources_ANA_different_input_strengths.png')



def with_vs_without_normalization():
    """
        load data of one pulse simulation
        demonstrate input+flow+BOLD for standard BOLD monitor with and without normalization
        additionally show same data without a pulse stimulus
    """
    ### LOAD DATA
    load_string_pulse = '5_0_1'
    load_string_rest = '1_0_1'
    recordingsB_pulse = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string_pulse+'.npy', allow_pickle=True).item()
    recordingsB_rest = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string_rest+'.npy', allow_pickle=True).item()
    simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string_pulse+'.npy', allow_pickle=True).item()

    times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])

    ### PLOT
    plt.figure(figsize=(16,9),dpi=500)

    ## WITHOUT NORMALIZATION
    normalization_plot_column(title='without normalization', mon_name='1withoutNorm', col=0, times=times, recordingsB_pulse=recordingsB_pulse, recordingsB_rest=recordingsB_rest)
    ## WITH NORMALIZATION
    normalization_plot_column(title='with normalization', mon_name='1', col=1, times=times, recordingsB_pulse=recordingsB_pulse, recordingsB_rest=recordingsB_rest)
    
    plt.tight_layout()
    plt.savefig('BOLDfromDifferentSources_ANA_with_vs_without_norm.png')



def pulses_visualization():
    """
        load data of one pulse simulation
        visualize CBF/CMRO2 and BOLD of the different BOLD monitors with different source signals
    """

    ### LOAD DATA
    load_string = '5_0_1'
    recordingsB = np.load('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+load_string+'.npy', allow_pickle=True).item()
    simParams   = np.load('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+load_string+'.npy', allow_pickle=True).item()

    times=np.arange(simParams['rampUp']+simParams['dt'],simParams['rampUp']+simParams['sim_dur']+simParams['dt'],simParams['dt'])


    ### PLOT
    plt.figure(figsize=(10,12),dpi=500)

    for row, label in enumerate(['A','B','C','D','E','F']):
        pulses_visualization_plot_row(row, label, recordingsB, times, simParams)

    plt.tight_layout()
    plt.savefig('BOLDfromDifferentSources_ANA_pulses_visualization.png', dpi=500)



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
    plt.savefig('BOLDfromDifferentSources_ANA_correlations_signals.png')

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
    plt.savefig('BOLDfromDifferentSources_ANA_correlations_matrix.png')
    

    


if __name__=='__main__':

    overview_plot=0
    different_input_strengths_plot=0
    with_vs_without_norm_plot=0
    pulses_visu_plot=0
    correlation_plot=1
    
    if overview_plot:
        if len(sys.argv)==3:
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
        with_vs_without_normalization()

    if pulses_visu_plot:
        pulses_visualization()

    if correlation_plot:
        BOLD_correlations()



















