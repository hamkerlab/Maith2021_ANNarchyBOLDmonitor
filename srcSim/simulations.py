from ANNarchy import *
from ANNarchy.extensions.bold import *
import pylab as plt
from model import params, rng, newBoldNeuron
from extras import getFiringRateDist, lognormalPDF, plot_input_and_raster, addMonitors, startMonitors, getMonitors


def initialTestofBOLD():
    #########################################   IMPORTANT SIMULATION PARAMS   ###########################################
    simParams={}
    simParams['dt']=params['dt']
    simParams['input']=params['input']
    simParams['rampUp']=1000#ms
    simParams['simDur']=10000#ms
    


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
                            input_variables="syn")
                                
    ### ALSO RECORD INPUT (r) OF BOLDNEURON
    monB['2'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            input_variables="syn",
                            recorded_variables=["BOLD", "r"])
                            
    ### SCALE THE POPULATION SIGNALS EQUALLY
    monB['3'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            scale_factor=[1,1],
                            input_variables="syn",
                            recorded_variables=["BOLD", "r"])
                            
    ### NORMALIZE THE POPULATION SIGNALS WITH BASELINE OVER 2000 ms
    monB['4'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            normalize_input=[2000,2000],
                            input_variables="syn",
                            recorded_variables=["BOLD", "r", "f_in"])
                            
    ### USE SELF DEFINED POPULATION SIGNALS (input_variables + BOLD_MODEL (output_variables + bold_model)
    monB['5'] = BoldMonitor(populations=[get_population('corEL1'), get_population('corIL1')],
                            normalize_input=[2000,2000],
                            input_variables=["var_f","var_r"],
                            output_variables=["I_f","I_r"],
                            bold_model=newBoldNeuron,
                            recorded_variables=["I_CBF","I_CMRO2","CBF","CMRO2","BOLD"])
    
    ### SELF-DEFINED ONLY corE WITHOUT NORMALIZATION
    monB['6'] = BoldMonitor(populations=get_population('corEL1'),
                            scale_factor=1,
                            input_variables=["var_f","var_r"],
                            output_variables=["I_f","I_r"],
                            bold_model=newBoldNeuron,
                            recorded_variables=["I_CBF","I_CMRO2","CBF","CMRO2","BOLD"])
    
    ### SELF-DEFINED ONLY corI WITHOUT NORMALIZATION
    monB['7'] = BoldMonitor(populations=get_population('corIL1'),
                            scale_factor=1,
                            input_variables=["var_f","var_r"],
                            output_variables=["I_f","I_r"],
                            bold_model=newBoldNeuron,
                            recorded_variables=["I_CBF","I_CMRO2","CBF","CMRO2","BOLD"])

    ### GENERATE monDict for BOLDMonitors, to easier start and get the monitors
    monDictB={'BOLD;1':['BOLD'],
              'BOLD;2':['BOLD', 'r'],
              'BOLD;3':['BOLD', 'r'],
              'BOLD;4':['BOLD', 'r', 'f_in'],
              'BOLD;5':["I_CBF","I_CMRO2","CBF","CMRO2","BOLD"],
              'BOLD;6':["I_CBF","I_CMRO2","CBF","CMRO2","BOLD"],
              'BOLD;7':["I_CBF","I_CMRO2","CBF","CMRO2","BOLD"]}



    ####################################################   COMPILE   ####################################################
    compile()

    ### INITIALIZE PARAMETERS OF OWN BOLD MODEL, kCBF from Friston
    kCBF = 1/2.46
    kCMRO2 = 2*kCBF
    monB['5'].k_CBF=kCBF
    monB['5'].k_CMRO2=kCMRO2
    monB['5'].c_CBF=0.6*np.sqrt(4*kCBF)
    monB['5'].c_CMRO2=np.sqrt(4*kCMRO2)
    


    ##################################################   SIMULATION   ###################################################


    ### RAMP_UP FOR MODEL
    simulate(simParams['rampUp'])


    ### START MONITORS

    ## standard monitors
    startMonitors(monDict,mon)

    ## BOLD monitors
    startMonitors(monDictB,monB)


    ### ACTUAL SIMULATION
    simulate(simParams['simDur'])


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


if __name__=='__main__':

    initialTestofBOLD()

