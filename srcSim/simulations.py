from ANNarchy import *
from ANNarchy.extensions.bold import *
import sys
import os
from model import params, rng, balloon_two_inputs, add_scaled_projections, BoldModel_r
from extras import getFiringRateDist, lognormalPDF, plot_input_and_raster, addMonitors, startMonitors, getMonitors


def initialTestofBOLD():
    #########################################   IMPORTANT SIMULATION PARAMS   ###########################################
    simParams={}
    simParams['dt']=params['dt']
    simParams['input']=params['input']
    simParams['rampUp']=1000#ms
    simParams['sim_dur']=30000#ms



    #########################################   ADD PROJECTIONS IF MODEL v2   ###########################################
    if 'v2' in params['optimizeRates']:
        """
            add the scaled projections of model v2
        """
        add_scaled_projections(params['fittedParams']['S_INP'], params['fittedParams']['S_EI'], params['fittedParams']['S_IE'], params['fittedParams']['S_II'], rng)


    ###################################################   MONITORS   ####################################################
    if params['input']=='Current':
        monDict={'pop;inputPop':['r'],
                 'pop;corEL1':['syn', 'spike'],
                 'pop;corIL1':['syn', 'spike']}
    elif params['input']=='Poisson':
        monDict={'pop;inputPop':['spike'],
                 'pop;corEL1':['syn', 'spike'],
                 'pop;corIL1':['syn', 'spike']}
    mon={}
    mon=addMonitors(monDict,mon)



    #################################################   BOLDMONITORS   ##################################################
    monB={}
    ### STANDARD BOLDMONITOR WITHOUT ANY OPTIONALS --> JUST DEFINE POPULATIONS AND INPUT VARIABLE
    monB['1'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            mapping={'I_CBF':'syn'})
                                
    ### ALSO RECORD INPUT (r) OF BOLDNEURON
    monB['2'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            mapping={'I_CBF':'syn'},
                            recorded_variables=["BOLD", "I_CBF"])
                            
    ### SCALE THE POPULATION SIGNALS EQUALLY
    monB['3'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            scale_factor=[1,1],
                            mapping={'I_CBF':'syn'},
                            recorded_variables=["BOLD", "I_CBF"])
                            
    ### NORMALIZE THE POPULATION SIGNALS WITH BASELINE OVER 2000 ms
    monB['4'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            normalize_input=[10000,10000],
                            mapping={'I_CBF':'syn'},
                            recorded_variables=["BOLD", "I_CBF", "f_in"])
                            
    ### USE SELF DEFINED POPULATION SIGNALS (source_variables + BOLD_MODEL (input_variables + bold_model)
    monB['5'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            normalize_input=[10000,10000],
                            mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                            bold_model=balloon_two_inputs,
                            recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
    
    ### SELF-DEFINED ONLY corE WITHOUT NORMALIZATION
    monB['6'] = BoldMonitor(populations=get_population('corEL1'),
                            scale_factor=1,
                            mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                            bold_model=balloon_two_inputs,
                            recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
    
    ### SELF-DEFINED ONLY corI WITHOUT NORMALIZATION
    monB['7'] = BoldMonitor(populations=get_population('corIL1'),
                            scale_factor=1,
                            mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                            bold_model=balloon_two_inputs,
                            recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
    
    ### ONLY corE WITH NORMALIZATION
    monB['8'] = BoldMonitor(populations=get_population('corEL1'),
                            scale_factor=1,
                            normalize_input=10000,
                            mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                            bold_model=balloon_two_inputs,
                            recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
    
    ### NLY corI WITH NORMALIZATION
    monB['9'] = BoldMonitor(populations=get_population('corIL1'),
                            scale_factor=1,
                            normalize_input=10000,
                            mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                            bold_model=balloon_two_inputs,
                            recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
                                
    ### Standard only corE
    monB['10'] = BoldMonitor(populations=get_population('corEL1'),
                            scale_factor=1,
                            mapping={'I_CBF':'syn'},
                            recorded_variables=["BOLD", "I_CBF"])
                                
    ### Standard only corI
    monB['11'] = BoldMonitor(populations=get_population('corIL1'),
                            scale_factor=1,
                            mapping={'I_CBF':'syn'},
                            recorded_variables=["BOLD", "I_CBF"])

    ### GENERATE monDict for BOLDMonitors, to easier start and get the monitors
    monDictB={'BOLD;1':['BOLD'],
              'BOLD;2':['BOLD', 'I_CBF'],
              'BOLD;3':['BOLD', 'I_CBF'],
              'BOLD;4':['BOLD', 'I_CBF', 'f_in'],
              'BOLD;5':["I_CBF","I_CMRO2","f_in","r","BOLD"],
              'BOLD;6':["I_CBF","I_CMRO2","f_in","r","BOLD"],
              'BOLD;7':["I_CBF","I_CMRO2","f_in","r","BOLD"],
              'BOLD;8':["I_CBF","I_CMRO2","f_in","r","BOLD"],
              'BOLD;9':["I_CBF","I_CMRO2","f_in","r","BOLD"],
              'BOLD;10':['BOLD', 'I_CBF'],
              'BOLD;11':['BOLD', 'I_CBF']}



    ####################################################   COMPILE   ####################################################
    compile()
    


    ##################################################   SIMULATION   ###################################################


    ### RAMP_UP FOR MODEL
    simulate(simParams['rampUp'])


    ### START MONITORS

    ## standard monitors
    startMonitors(monDict,mon)

    ## BOLD monitors
    startMonitors(monDictB,monB)


    ### ACTUAL SIMULATION
    simulate(simParams['sim_dur'])


    ### GET MONITORS

    ## standard monitors
    recordings={}
    recordings=getMonitors(monDict,mon,recordings)

    ## BOLD monitors
    recordingsB={}
    recordingsB=getMonitors(monDictB,monB,recordingsB)
    

    ### SAVE DATA
    np.save('../dataRaw/simulations_initialTestofBOLD_recordings.npy',recordings)
    np.save('../dataRaw/simulations_initialTestofBOLD_recordingsB.npy',recordingsB)
    np.save('../dataRaw/simulations_initialTestofBOLD_simParams.npy',simParams)



def BOLDfromDifferentSources(input_factor=1.0, stimulus=0, simID='', monitoring=0):
    """
        input_factor: float
            change of input during input pulses, default=1.0
        stimulus: int
            0: default, long input change (one time change)
            1: short input pulse
            2: long resting state where only BOLD is recorded
            3: long input pulse
        simID: string
            simulation identificator (e.g. number), default=''
        monitoring: int
            0: default, all monitors active
            1: no monitor activate
            2: only a single BOLD monitor active
            
    """
    #########################################   IMPORTANT SIMULATION PARAMS   ###########################################
    simParams={}
    for key in ['dt', 'input', 'corE_popsize', 'seed']:
        simParams[key]=params[key]
    simParams['rampUp']=2000#ms !!!ATTENTION!!! rampUp has to be >= firingRateWindow otherwise firing rate starts at a wrong value
    simParams['stimulus']=stimulus
    if stimulus==0:
        ## long input
        simParams['sim_dur1'], simParams['sim_dur2'], simParams['sim_dur3']= [5000,20000,0]#ms
    elif stimulus==1:
        ## short input pulse
        simParams['sim_dur1'], simParams['sim_dur2'], simParams['sim_dur3']= [10000,100,14900]#ms
    elif stimulus==2:
        ## long resting period where only BOLD is recorded
        simParams['sim_dur1'], simParams['sim_dur2'], simParams['sim_dur3']= [10*60*1000,0,0]#ms
    elif stimulus==3:
        ## long impulse
        simParams['sim_dur1'], simParams['sim_dur2'], simParams['sim_dur3'] = [5000,20000,15000]#ms
    else:
        print('second argument, stimulus, has to be 0 or 1')
        quit()
    simParams['sim_dur']=simParams['sim_dur1'] + simParams['sim_dur2'] + simParams['sim_dur3']
    simParams['BOLDbaseline']=5000#ms
    simParams['firingRateWindow']=20#ms
    simParams['input_factor']=input_factor
    simParams['monitoring']=monitoring
    save_string = str(simParams['input_factor']).replace('.','_')+'_'+str(simParams['stimulus']).replace('.','_')
    if len(simID)>0:
        save_string=save_string+'__'+simID
    ## append saving files if monitoring or populations size is different from default
    if monitoring!=0:
        save_string=save_string+'__'+str(int(monitoring))
    if params['corE_popsize']!=params['corE_popsize_default']:
        save_string=save_string+'__'+str(int(params['corE_popsize']))
        


    #########################################   ADD PROJECTIONS IF MODEL v2   ###########################################
    if 'v2' in params['optimizeRates']:
        """
            add the scaled projections of model v2
        """
        add_scaled_projections(params['fittedParams']['S_INP'], params['fittedParams']['S_EI'], params['fittedParams']['S_IE'], params['fittedParams']['S_II'], rng)


    ###################################################   MONITORS   ####################################################
    if stimulus!=2:
        ### DEACTIVATE MONITORS FOR LONG RESTING SIMULATION
        if params['input']=='Current':
            monDict={'pop;inputPop':['r'],
                     'pop;corEL1':['syn', 'spike'],
                     'pop;corIL1':['syn', 'spike']}
        elif params['input']=='Poisson':
            monDict={'pop;inputPop':['spike'],
                     'pop;corEL1':['syn', 'spike', 'var_r', 'var_ra', 'r'],
                     'pop;corIL1':['syn', 'spike', 'var_r', 'var_ra', 'r']}
        if monitoring!=0:
            monDict={}
        mon={}
        mon=addMonitors(monDict,mon)



    #################################################   BOLDMONITORS   ##################################################
    """
        Generate BOLD monitors with different input signals.
        Additionally implement BOLD monitors for the individual populations, without normalization to verify the raw input signals.
        Attention! For the raw input signals, the BOLD calculation may "explode" --> Use BOLD model (BoldModel_r) which doesn't perform calculations
    """

    monB={}
    if monitoring==0:
        ### BOLD monitor which uses single input: all syn input
        monB['1'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                normalize_input=[simParams['BOLDbaseline'],simParams['BOLDbaseline']],
                                mapping={'I_CBF':'syn'},
                                recorded_variables=["BOLD", "I_CBF", "f_in", "E", "q", "v", "f_out"])
        monB['1withoutNorm'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                           mapping={'I_CBF':'syn'},
                                           recorded_variables=["BOLD", "I_CBF", "f_in"])
        monB['1Eraw'] = BoldMonitor(populations=get_population('corEL1'),
                                    scale_factor=1,
                                    mapping={'I_CBF':'syn'},
                                    bold_model=BoldModel_r,
                                    recorded_variables=["I_CBF"])
        monB['1Iraw'] = BoldMonitor(populations=get_population('corIL1'),
                                    scale_factor=1,
                                    mapping={'I_CBF':'syn'},
                                    bold_model=BoldModel_r,
                                    recorded_variables=["I_CBF"])
                                
        ### single input: excitatory syn input
        monB['2'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                normalize_input=[simParams['BOLDbaseline'],simParams['BOLDbaseline']],
                                mapping={'I_CBF':'g_ampa'},
                                recorded_variables=["BOLD", "I_CBF", "f_in", "E", "q", "v", "f_out"])
        monB['2Eraw'] = BoldMonitor(populations=get_population('corEL1'),
                                    scale_factor=1,
                                    mapping={'I_CBF':'g_ampa'},
                                    bold_model=BoldModel_r,
                                    recorded_variables=["I_CBF"])
        monB['2Iraw'] = BoldMonitor(populations=get_population('corIL1'),
                                    scale_factor=1,
                                    mapping={'I_CBF':'g_ampa'},
                                    bold_model=BoldModel_r,
                                    recorded_variables=["I_CBF"])
        
        ### single input: firing rate
        ## to record firing rate from spiking populations it has to be calculated
        get_population('corEL1').compute_firing_rate(window=simParams['firingRateWindow'])
        get_population('corIL1').compute_firing_rate(window=simParams['firingRateWindow'])

        monB['3'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                normalize_input=[simParams['BOLDbaseline'],simParams['BOLDbaseline']],
                                mapping={'I_CBF':'r'},
                                recorded_variables=["BOLD", "I_CBF", "f_in", "E", "q", "v", "f_out"])
        monB['3Eraw'] = BoldMonitor(populations=get_population('corEL1'),
                                    scale_factor=1,
                                    mapping={'I_CBF':'r'},
                                    bold_model=BoldModel_r,
                                    recorded_variables=["I_CBF"])
        monB['3Iraw'] = BoldMonitor(populations=get_population('corIL1'),
                                    scale_factor=1,
                                    mapping={'I_CBF':'r'},
                                    bold_model=BoldModel_r,
                                    recorded_variables=["I_CBF"])
        
        ### two inputs: Buxton (2012, 2014, 2021), excitatory --> CMRO2&CBF, inhibitory --> CBF, use post-synaptic currents as driving signals (they likely cause metabolism)
        monB['4'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                normalize_input=[simParams['BOLDbaseline'],simParams['BOLDbaseline']],
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                                bold_model=balloon_two_inputs,
                                recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
        monB['4Eraw'] = BoldMonitor(populations=get_population('corEL1'),
                                scale_factor=1,
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                                bold_model=BoldModel_r,
                                recorded_variables=["I_CBF","I_CMRO2"])
        monB['4Iraw'] = BoldMonitor(populations=get_population('corIL1'),
                                scale_factor=1,
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_r'},
                                bold_model=BoldModel_r,
                                recorded_variables=["I_CBF","I_CMRO2"])
        
        ### two inputs: Buxton + Howarth et al. (2021), in interneurons firing rate drives CMRO2
        monB['5'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                normalize_input=[simParams['BOLDbaseline'],simParams['BOLDbaseline']],
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_ra'},
                                bold_model=balloon_two_inputs,
                                recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
        monB['5Eraw'] = BoldMonitor(populations=get_population('corEL1'),
                                scale_factor=1,
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_ra'},
                                bold_model=BoldModel_r,
                                recorded_variables=["I_CBF","I_CMRO2"])
        monB['5Iraw'] = BoldMonitor(populations=get_population('corIL1'),
                                scale_factor=1,
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_ra'},
                                bold_model=BoldModel_r,
                                recorded_variables=["I_CBF","I_CMRO2"])
        
        ### two inputs: Buxton, but flow is quadratic
        monB['6'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                normalize_input=[simParams['BOLDbaseline'],simParams['BOLDbaseline']],
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_rb'},
                                bold_model=balloon_two_inputs,
                                recorded_variables=["I_CBF","I_CMRO2","f_in","r","BOLD"])
        monB['6Eraw'] = BoldMonitor(populations=get_population('corEL1'),
                                scale_factor=1,
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_rb'},
                                bold_model=BoldModel_r,
                                recorded_variables=["I_CBF","I_CMRO2"])
        monB['6Iraw'] = BoldMonitor(populations=get_population('corIL1'),
                                scale_factor=1,
                                mapping={'I_CBF':'var_f', 'I_CMRO2':'var_rb'},
                                bold_model=BoldModel_r,
                                recorded_variables=["I_CBF","I_CMRO2"])

        ### GENERATE monDict for BOLDMonitors, to easier start and get the monitors
        monDictB={'BOLD;1':            ['BOLD', 'I_CBF', "f_in", "E", "q", "v", "f_out"],
                  'BOLD;1withoutNorm': ['BOLD', 'I_CBF', "f_in"],
                  'BOLD;1Eraw':        ['I_CBF'],
                  'BOLD;1Iraw':        ['I_CBF'],
                  'BOLD;2':            ['BOLD', 'I_CBF', "f_in", "E", "q", "v", "f_out"],
                  'BOLD;2Eraw':        ['I_CBF'],
                  'BOLD;2Iraw':        ['I_CBF'],
                  'BOLD;3':            ['BOLD', 'I_CBF', "f_in", "E", "q", "v", "f_out"],
                  'BOLD;3Eraw':        ['I_CBF'],
                  'BOLD;3Iraw':        ['I_CBF'],
                  'BOLD;4':            ["I_CBF","I_CMRO2","f_in","r","BOLD"],
                  'BOLD;4Eraw':        ["I_CBF","I_CMRO2"],
                  'BOLD;4Iraw':        ["I_CBF","I_CMRO2"],
                  'BOLD;5':            ["I_CBF","I_CMRO2","f_in","r","BOLD"],
                  'BOLD;5Eraw':        ["I_CBF","I_CMRO2"],
                  'BOLD;5Iraw':        ["I_CBF","I_CMRO2"],
                  'BOLD;6':            ["I_CBF","I_CMRO2","f_in","r","BOLD"],
                  'BOLD;6Eraw':        ["I_CBF","I_CMRO2"],
                  'BOLD;6Iraw':        ["I_CBF","I_CMRO2"]}
    elif monitoring==1:
        monDictB={}
    else:
        monB['1'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                                normalize_input=[simParams['BOLDbaseline'],simParams['BOLDbaseline']],
                                mapping={'I_CBF':'syn'},
                                recorded_variables=["BOLD", "I_CBF", "f_in", "E", "q", "v", "f_out"])
        monDictB={'BOLD;1':            ['BOLD', 'I_CBF', "f_in", "E", "q", "v", "f_out"]}



    ####################################################   COMPILE   ####################################################
    compile('annarchy_folders/annarchy_'+save_string)
    if os.getcwd().split('/')[-1]=='annarchy_folders': os.chdir('../')

    


    ##################################################   SIMULATION   ###################################################


    ### RAMP_UP FOR MODEL
    simulate(simParams['rampUp'])


    ### START MONITORS

    ## standard monitors
    if stimulus!=2:
        ## DEACTIVATE MONITORS FOR LONG RESTING SIMULATION
        startMonitors(monDict,mon)

    ## BOLD monitors
    startMonitors(monDictB,monB)

    ## Start performance measurement (if requested)
    if params["measure_time"]:
        t1 = time.time()

    ### ACTUAL SIMULATION
    simulate(simParams['sim_dur1'])
    ## increase input
    get_population('inputPop').offsetVal = params['inputPop_init_offsetVal']*simParams['input_factor']
    simulate(simParams['sim_dur2'])
    ## reset input
    get_population('inputPop').offsetVal = params['inputPop_init_offsetVal']
    simulate(simParams['sim_dur3'])

    ## Stop performance measurement (if requested)
    if params["measure_time"]:
        t2 = time.time()
        if monitoring==1:
            with open("../dataRaw/perf_without_monitor_"+str(params['corE_popsize'])+"_"+str(params['numInputs'])+"_"+str(params['num_threads'])+"threads.csv", "a") as file:
                file.write(str(t2-t1)+"\n")
        elif monitoring==2:
            with open("../dataRaw/perf_with_monitor_"+str(params['corE_popsize'])+"_"+str(params['numInputs'])+"_"+str(params['num_threads'])+"threads.csv", "a") as file:
                file.write(str(t2-t1)+"\n")
        else:
            pass

    ### GET MONITORS

    ## standard monitors
    if stimulus!=2:
        recordings={}
        recordings=getMonitors(monDict,mon,recordings)

    ## BOLD monitors
    recordingsB={}
    recordingsB=getMonitors(monDictB,monB,recordingsB)
    

    ### SAVE DATA
    if stimulus!=2: np.save('../dataRaw/simulations_BOLDfromDifferentSources_recordings_'+save_string+'.npy',recordings)
    np.save('../dataRaw/simulations_BOLDfromDifferentSources_recordingsB_'+save_string+'.npy',recordingsB)
    np.save('../dataRaw/simulations_BOLDfromDifferentSources_simParams_'+save_string+'.npy',simParams)




if __name__=='__main__':

    if len(sys.argv)==5:
        ## optional input_factor, stimulus, simulation ID and monitoring mode given
        BOLDfromDifferentSources(input_factor=float(sys.argv[1]), stimulus=int(sys.argv[2]), simID=str(int(sys.argv[3])), monitoring=int(sys.argv[4]))
    elif len(sys.argv)==4:
        ## optional input_factor, stimulus and simulation ID given
        BOLDfromDifferentSources(input_factor=float(sys.argv[1]), stimulus=int(sys.argv[2]), simID=str(int(sys.argv[3])))
    elif len(sys.argv)==3:
        ## optional input_factor and stimulus given
        BOLDfromDifferentSources(input_factor=float(sys.argv[1]), stimulus=int(sys.argv[2]))
    elif len(sys.argv)==2:
        ## optional input_factor given
        BOLDfromDifferentSources(input_factor=float(sys.argv[1]))
    else:
        ## standard simulation arguments used
        BOLDfromDifferentSources()

