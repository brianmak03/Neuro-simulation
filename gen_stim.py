import numpy as np
from collections import namedtuple
import numpy.linalg as LA
import itertools as it
import pandas as pd
import pickle

distribuiton = namedtuple('distribution', 'label, mean, covariance')

def generate_stimuli(mean, covariance, n):

    dimensionality = len(mean)
    X_old = np.random.multivariate_normal(np.zeros(dimensionality), np.eye(dimensionality), n).T
    Z = ((LA.inv(LA.cholesky(np.cov(X_old)))@X_old).T - np.mean(LA.inv(LA.cholesky(np.cov(X_old)))@X_old, axis=1)).T
    X_new = (LA.cholesky(covariance)@Z).T + mean
    return X_new

grid = tuple(it.product(range(50), repeat=2))

cat0 = distribuiton(0, [18.75, 25], np.array([[25, 0],[0, 25]]))
cat1 = distribuiton(1, [25, 32], np.array([[25, 0],[0, 25]]))
cat2 = distribuiton(2, [25, 18.75], np.array([[25, 0],[0, 25]]))
cat3 = distribuiton(3, [32, 25], np.array([[25, 0],[0, 25]]))

categories = [cat0,cat1,cat2,cat3]

if __name__ == '__main__':

    stimuli = []
    for i in range(3):

        processed = pd.DataFrame(columns=['label','index','I'])
        for cat in categories:

            raw_stim = generate_stimuli(cat.mean, cat.covariance, 225)
            
            for stim in raw_stim:
                d = np.square(stim - grid).sum(dtype=np.float32, axis=1)
                I = np.exp(-d/4.5)
                index = np.where(I>0.001)
                I = I[index]
                I = I[np.newaxis]
                processed = processed.append({'label':cat.label,'index':index,'I':I}, ignore_index=True)

        processed = processed.sample(frac=1).reset_index(drop=True)
        random_feedback = np.concatenate(((np.zeros(75), np.ones(225))))
        np.random.shuffle(random_feedback)
        random = pd.DataFrame({'random':random_feedback}, index=np.arange(300,600))
        processed = pd.concat([processed,random],axis=1)
        stimuli.append(processed)
    
    pickling_on = open('stimuli.pickle', 'wb')
    pickle.dump(stimuli, pickling_on)
    pickling_on.close()