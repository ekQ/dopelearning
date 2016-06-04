"""
Code for analyzing (anonymized) usage logs from deepbeat.org and reproducing
Figure 4 from "Malmi, E. et al. DopeLearning: A Computational Approach to Rap
Lyrics Generation. In Proc. KDD, 2016."

Usage:
    python analyze_logs.py
    (See functions main() and analyze() below for options.)
"""
import json
import gzip
import datetime as dt
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import os


def read_log(fname):
    selections = json.load(gzip.open(fname,'rb'))
    users = [x['userID'] for x in selections]
    return users, selections

def get_good_selections(us, n_first_removed):
    good_selections = []
    n_users = 0
    for ss in us.itervalues():
        good_selections += ss[n_first_removed:]
        if len(ss) > n_first_removed:
            n_users += 1
    print "{} users with at least {} selections.".format(n_users, n_first_removed+1)
    return good_selections

def extract_feedback(us, n_first_removed, nn):
    print "Extracting feedback..."
    good_selections = get_good_selections(us, n_first_removed)
    score_differences = []
    for s in good_selections:
        sel_line = s['selectedLine']
        if nn:
            scores = [lc['nn_score'] for lc in s['lineCandidates']]
        else:
            scores = [lc['score'] for lc in s['lineCandidates']]
        bl_score = scores[sel_line]
        for score in scores[:sel_line]:
            diff = bl_score - score
            score_differences.append(diff)
    print "Extracted %d pairwise preferences." % len(score_differences)
    return np.asarray(score_differences)

def compute_probs(score_differences, max_bin, bin_width=0.5):
    fb = np.asarray(score_differences)
    bins = np.arange(0, max_bin+0.01, bin_width)
    
    probs = []
    bin_centers = []
    stds = []    # Standard deviations
    for i in range(len(bins)-1):
        lb = bins[i]
        ub = bins[i+1]
        n_good = sum(np.logical_and(fb>=lb, fb<ub))
        n_bad = sum(np.logical_and(fb<=-lb, fb>-ub))
        n_all = n_good + n_bad
        p = float(n_good) / (n_all)
        std = np.sqrt(p * (1 - p) / n_all)
        print "Bin lb={}, ub={}, n={}, p={:.3f}, std={:.3f}".format(
            bins[i], bins[i+1], n_all, p, std)
        stds.append(std)
        probs.append(p)
        bin_centers.append((ub+lb) / 2.0)
    return bin_centers, probs, stds

def analyze(log_fname, nn=True, max_bin=6, n_first_removed=3):
    """
    Analyze anonymized deepbeat.org usage logs.
    
    Input:
        log_fname -- Path to the log file.
        nn -- Use scores with the NN feature or not.
        max_bin -- Maximum score difference considered
                   (if too large, the last bins get very noisy).
        n_first_removed -- From each user, remove this many first selections,
                           since in the beginning the user might be just
                           playing with the tool.       
    Output:
        bin_centers
        probabilities to select the better line according to the algorithm
        standard deviations
    """
    users, selections = read_log(log_fname)
    print "%d selections, %d unique users" % (len(selections), len(set(users)))
    us = {}    # user -> selections
    for u, s in zip(users, selections):
        if u not in us:
            us[u] = []
        us[u].append(s)
    lens = [len(s) for s in us.itervalues()]
    lens = np.array(lens)
    
    score_differences = extract_feedback(us, n_first_removed, nn)
    print "t-test:", stats.ttest_1samp(score_differences, 0)

    sel_ranks = [s['selectedLine'] for s in selections]
    print "Histogram of selected line indices:"
    print np.histogram(sel_ranks, range(21))

    xs, probs, stds = compute_probs(score_differences, max_bin)
    return xs, probs, stds

def main():
    # Compute probabilities
    log_fname = os.path.join('..', 'usage_logs',
                             'selected_lines_anonymized.json.gz')
    n_first_removed = 3
    max_bin = 5.5
    nn = True
    print "--- With NN ---"
    xs_nn, probs_nn, stds_nn = analyze(log_fname, nn, max_bin, n_first_removed)
    nn = False
    print "\n--- Without NN ---"
    xs, probs, stds = analyze(log_fname, nn, max_bin, n_first_removed)
    
    # Plot results
    index = np.arange(len(probs))
    bar_width = 0.4
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    plt.close()
    plt.figure(figsize=(7.5, 5))
    rects1 = plt.bar(index, probs_nn, bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=stds_nn,
                     error_kw=error_config,
                     label='DeepBeat')

    rects2 = plt.bar(index+bar_width, probs, bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=stds,
                     error_kw=error_config,
                     label='DeepBeat without NN')

    yspace = 0.02
    plt.ylim(min(probs + probs_nn)-yspace,
             max(probs + probs_nn)+max(stds+stds_nn)+0.01)
    plt.xlabel('score(line1) - score(line2)', fontsize=16)
    plt.ylabel('Probability to select line1', fontsize=16)
    # Tick labels
    labs = []
    xs = np.arange(0, 10, 0.5)
    for i in range(len(xs)-1):
        if i % 2 == 1:
            l = "{:.1f}-{:d}".format(xs[i],int(xs[i+1]))
        else:
            l = "{:d}-{:.1f}".format(int(xs[i]),xs[i+1])
        labs.append(l)
    plt.xticks(index + bar_width, labs)
    plt.xlim(0, 2*max_bin)
    plt.legend(loc=2)

    plt.tight_layout()
    out_fname = 'deepbeat_experiment_bar_anonymized.pdf'
    plt.savefig(out_fname)
    print "Stored the plot to:", out_fname
    plt.show()

if __name__ == '__main__':
    main()

