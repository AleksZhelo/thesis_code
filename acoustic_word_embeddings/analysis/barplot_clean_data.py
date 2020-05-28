import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


def _autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


def __plot_margin_clean_vs_patient():
    #  Val AP &      Test AP & Val accuracy & Test accuracy
    margin_mean, margin_std = (0.878, 0.863, 0.891, 0.897), (0.002, 0.002, 0.002, 0.003)
    margin_patient_mean, margin_patient_std = (0.525, 0.514, 0.726, 0.724), (0.009, 0.009, 0.006, 0.008)

    ind = np.arange(len(margin_mean))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(ind - width / 2, margin_mean, width, yerr=margin_std,
                    color='#396AB1', label='Margin loss clean', alpha=1.0)
    rects2 = ax.bar(ind + width / 2, margin_patient_mean, width, yerr=margin_patient_std,
                    color='#CC2529', label='Margin loss patient', alpha=1.0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(ind)
    ax.set_xticklabels(('Val AP', 'Test AP', 'Val accuracy', 'Test accuracy'))
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
              fancybox=True, shadow=True, ncol=2)

    _autolabel(ax, rects1, "left")
    _autolabel(ax, rects2, "right")

    # plt.show()
    plt.savefig('margin_clean_vs_patient.pdf', dpi=300, bbox_inches='tight', pad_inches=0)


def __plot_classifier_clean_vs_patient():
    #  Val AP &      Test AP & Val accuracy & Test accuracy
    classifier_mean, classifier_std = (0.655, 0.595, 0.811, 0.807), (0.011, 0.011, 0.002, 0.004)
    classifier_patient_mean, classifier_patient_std = (0.309, 0.287, 0.650, 0.662), (0.009, 0.002, 0.007, 0.014)

    ind = np.arange(len(classifier_mean))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(ind - width / 2, classifier_mean, width, yerr=classifier_std,
                    color='#396AB1', label='Classifier clean', alpha=1.0)
    rects2 = ax.bar(ind + width / 2, classifier_patient_mean, width, yerr=classifier_patient_std,
                    color='#CC2529', label='Classifier patient', alpha=1.0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(ind)
    ax.set_xticklabels(('Val AP', 'Test AP', 'Val accuracy', 'Test accuracy'))
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
              fancybox=True, shadow=True, ncol=2)

    _autolabel(ax, rects1, "left")
    _autolabel(ax, rects2, "right")

    plt.savefig('classifier_clean_vs_patient.pdf', dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    font = {'family': 'serif', 'serif': ['cmr10']}
    rc('font', **font)

    __plot_margin_clean_vs_patient()
    __plot_classifier_clean_vs_patient()
