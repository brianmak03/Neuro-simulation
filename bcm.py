from SPC import SPC
import numpy as np
import numpy.linalg as LA
import pandas as pd
import itertools as it
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation

distribuiton = namedtuple('distribution', 'label, mean, covariance')

phases = ['aquisition', 'intervention', 'reaquisition']

def generate_stimuli(mean, covariance, n):

    dimensionality = len(mean)
    X_old = np.random.multivariate_normal(np.zeros(dimensionality), np.eye(dimensionality), n).T
    Z = ((LA.inv(LA.cholesky(np.cov(X_old)))@X_old).T - np.mean(LA.inv(LA.cholesky(np.cov(X_old)))@X_old, axis=1)).T
    X_new = (LA.cholesky(covariance)@Z).T + mean
    return X_new


def score(response, label):

    if label is None:

        return 0

    return int(response == label) - int(response != label)


def simulate(categories, condition, random_seed=None, BCM=False, get_weights=False):

    np.random.seed(random_seed)

    sensory_cortex_units = tuple(it.product(range(200), repeat=2))
    striatal_units = [x.label for x in categories]

    respondent = SPC(sensory_cortex_units, striatal_units, BCM)

    stimuli = []
    for category in categories:

        temp = pd.DataFrame(generate_stimuli(category.mean, category.covariance, 225))
        temp['label'] = category.label
        stimuli.append(temp)

    stimuli = pd.concat(stimuli, ignore_index=True)
    stimuli = stimuli.sample(frac=1).reset_index(drop=True)
    stimuli, labels = np.split(stimuli.values, [-1], axis=1)
    labels = labels.ravel()

    NMDA = []
    accuracy = []
    record = []

    weights = [[],[],[],[]]

    #aquisition trials
    for i in range(300):

        response = respondent.simulate(stimuli[i])
        feedback = score(response, labels[i])

        NMDA.append(respondent.update_weights(feedback))

        if get_weights:

            weights[0].append(np.resize(respondent.W[0], (200,200)))
            weights[1].append(np.resize(respondent.W[1], (200,200)))
            weights[2].append(np.resize(respondent.W[2], (200,200)))
            weights[3].append(np.resize(respondent.W[3], (200,200)))

        record.append(response==labels[i])

        if (i+1) % 25 == 0:
            accuracy.append(np.mean(record))
            record = []


    #intervention trials
    random_feedback = list(np.concatenate(((np.ones(75), -1*np.ones(225)))))
    np.random.shuffle(random_feedback)

    for i in range(300, 600):

        response = respondent.simulate(stimuli[i])

        NMDA.append(respondent.update_weights(random_feedback.pop()))

        if get_weights:

            weights[0].append(np.resize(respondent.W[0], (200,200)))
            weights[1].append(np.resize(respondent.W[1], (200,200)))
            weights[2].append(np.resize(respondent.W[2], (200,200)))
            weights[3].append(np.resize(respondent.W[3], (200,200)))

        record.append(response==labels[i])

        if (i+1) % 25 == 0:
            accuracy.append(np.mean(record))
            record = []

    
    if condition == 'meta':
        metalabels = {categories[i].label: categories[i-1].label for i in range(len(categories))}
        labels[600:] = [metalabels[label] for label in labels[600:]]

    #reaquisition trails
    for i in range(600, 900):

        response = respondent.simulate(stimuli[i])
        feedback = score(response, labels[i])
        
        NMDA.append(respondent.update_weights(feedback))

        if get_weights:

            weights[0].append(np.resize(respondent.W[0], (200,200)))
            weights[1].append(np.resize(respondent.W[1], (200,200)))
            weights[2].append(np.resize(respondent.W[2], (200,200)))
            weights[3].append(np.resize(respondent.W[3], (200,200)))

        record.append(response==labels[i])

        if (i+1) % 25 == 0:
            accuracy.append(np.mean(record))
            record = []

    if get_weights == True:
        
        return accuracy, list(zip(*NMDA)), weights

    return accuracy, list(zip(*NMDA))


def plot_heatmap(w, lr, nt, title):

    fig = plt.figure()
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(223)
    ax6 = fig.add_subplot(224)
    axs = [ax1, ax2, ax3, ax4]

    reel = []
    for i in range(len(w[0])):

        if i >= 600:
            phase = phases[2]

        elif i >= 300:
            phase = phases[1]

        else:
            phase = phases[0]

        frames = []
        for j in range(len(axs)):

            frame = axs[j].imshow(w[j][i], vmin=0, vmax=1, cmap='coolwarm')
            text = axs[j].annotate(phase, xy=(1,1), xytext=(10,-10), textcoords='offset points')
            frames.append(frame)
            frames.append(text)

        reel.append(frames)

    ani = animation.ArtistAnimation(fig, reel, interval=10, blit=True)

    ax5.plot(lr)
    ax5.set_title('Learning_rate')
    ax5.set_xticks([0,12,24])
    ax5.set_xticklabels(phases)
    for i in range(len(nt)):
        ax6.plot(nt[i])
    ax6.set_title('NMDA Threshold')
    ax6.set_xticks([0,300,600])
    ax6.set_xticklabels(phases)

    fig.suptitle(title)
    plt.show()


def plot_avg(lr, nt, title):

    fig, axs = plt.subplots(2,1)
    axs[0].plot(lr)
    axs[0].set_ylabel('Learning Rate')
    axs[0].set_xticks([0,12,24])
    axs[0].set_xticklabels(phases)
    for i in range(len(nt)):
        axs[1].plot(nt[i])
    axs[1].set_ylabel('NMDA Threshold')
    axs[1].set_xticks([0,300,600])
    axs[1].set_xticklabels(phases)
    fig.suptitle(title)
    plt.show()



if __name__ == '__main__':

    A = distribuiton('A', [72, 100], np.array([[100, 0],[0, 100]]))
    B = distribuiton('B', [100, 128], np.array([[100, 0],[0, 100]]))
    C = distribuiton('C', [100, 72], np.array([[100, 0],[0, 100]]))
    D = distribuiton('D', [128, 100], np.array([[100, 0],[0, 100]]))

    categories = [A,B,C,D]

    # learning_rate = []
    # NMDA_thresh = []

    # for _ in range(33):

    #     accuracy, NMDA = simulate(categories, condition='meta', BCM=False)
    #     learning_rate.append(accuracy)
    #     NMDA_thresh.append(NMDA)

    # learning_rate = np.mean(learning_rate, axis=0)
    # NMDA_thresh = np.mean(NMDA_thresh, axis=0)

    # plot_avg(learning_rate, NMDA_thresh, 'Meta-Learning')


    accuracy, NMDA, weights = simulate(categories, condition='meta', BCM=True, get_weights=True)
    plot_heatmap(weights, accuracy, NMDA, 'Meta-Learning')