import numpy as np
import pandas as pd
import pickle

def simulate(a,b):

    pickling_off = open("stimuli.pickle","rb")
    stimuli = pickle.load(pickling_off)
    pickling_off.close()
    np.random.seed(99)

    acc = []
    for stimulus in stimuli:

        #initialization of weights
        W = np.random.uniform(0.01, 0.2, (4,2500))

        #initialize previous obtained and predicted rewards
        R_prev = 0.25
        P_prev = 0.25

        #initialize record and accuracy, accuracy averaged each 15 trial block
        bloc_acc = []
        record = []

        #start trials. Go through each stimulus
        for stim in stimulus.itertuples():

            index = stim.index
            I = stim.I
            label = stim.label
            #calculate S using only the entries in I that are over 0.001
            S = W[:,index]@I.T
            #response is most activated straital unit
            response = np.argmax(S)
            #All straital neurons inhibit each other except the most activated does not get inhibited
            inhibit = np.sum([np.concatenate([S[i+1:], S[:i+1]]) for i in range(3)], axis=0)
            inhibit[response] = 0
            S = S - 0.1*inhibit #+ 0.1*np.random.normal(0, 1, size=(4,1))
            #obtained reward, random for 75% or intervention trials
            if 299 < stim.Index < 600 and stim.random:
                R = np.random.choice([-1,1], p=[0.75, 0.25])
            else:
                R = int(response==label) - int(response!=label)
            #predicted reward
            P = P_prev + 0.025*(R_prev - P_prev)
            RPE = R - P
            #calculate dopamine release
            DA = (RPE >= -0.25)*np.minimum(1, 0.8*RPE+0.2)
            #update relavent weights
            W[:,index] = (W[:,index] + a*(np.maximum(S - 0.1, 0)@I)*np.maximum(DA - 0.2, 0)*(1 - W[:,index])
                            - b*(np.maximum(S - 0.1, 0)@I)*np.maximum(0.2 - DA, 0)*W[:,index])
            #record answer
            record.append(response==label)
            #average every 15th trial
            if (stim.Index+1) % 25 == 0:
                bloc_acc.append(np.mean(record))
                record = []
            R_prev = R
            P_prev = P
        
        acc.append(bloc_acc)

    acc = np.mean(acc, axis=0)

    return acc

