import numpy as np
import pandas as pd

from glob import glob
from sklearn.metrics import cohen_kappa_score


annot2class = {"Incorrect": 0, "I Don't Know": 1, "Correct": 2}


def main():

    ######################
    # EXPERT ANNOTATIONS #
    ######################

    print('Expert annotations statistics:')
    # read expert annotations
    expert1 = pd.read_csv('../data/annotations/experts/expert1/annotations.csv', keep_default_na=False)
    expert2 = pd.read_csv('../data/annotations/experts/expert2/annotations.csv', keep_default_na=False)
    expert3 = pd.read_csv('../data/annotations/experts/expert3/annotations.csv', keep_default_na=False)  # expert3 represents the tie breaker

    # convert annotation DataFrames to dicts
    annot1 = {(row['Subject'], row['Predicate'], row['Object']): row['Annotation'] for ix, row in expert1.iterrows()}
    annot2 = {(row['Subject'], row['Predicate'], row['Object']): row['Annotation'] for ix, row in expert2.iterrows()}
    annot3 = {(row['Subject'], row['Predicate'], row['Object']): row['Annotation'] for ix, row in expert3.iterrows()}

    # sort expert1 and expert2 annotations to compare them
    annot1 = dict(sorted(annot1.items()))
    annot2 = dict(sorted(annot2.items()))

    # count facts in disagreement between expert1 and expert2
    disagreement = sum([1 if annot1[k] != annot2[k] else 0 for k in annot1.keys()])
    print(f'Disagreement for {disagreement} facts out of {len(annot1)} ({round(disagreement/len(annot1)*100)}%)')

    # compute kappa score between expert1 and expert2
    classes1 = [annot2class[annot] for annot in annot1.values()]
    classes2 = [annot2class[annot] for annot in annot2.values()]
    k_score = cohen_kappa_score(classes1, classes2, labels=[0, 1, 2], weights='linear')
    print(f"Cohen's kappa score: {round(k_score, 2)}")

    # set counter of unresolved ties
    unresolved = 0

    # set ground truth var
    gt = {}
    # generate expert-based ground truth
    for fact in annot1.keys():
        # store fact annotations
        if annot1[fact] == annot2[fact]:  # agreement between experts
            gt[fact] = annot1[fact]
        else:  # disagreement -- break ties w/ expert3
            # set annotation counter
            annot2count = {'Correct': 0, 'Incorrect': 0, "I Don't Know": 0}
            # store expert annotations
            annot2count[annot1[fact]] += 1
            annot2count[annot2[fact]] += 1
            annot2count[annot3[fact]] += 1

            # aggregate annotations via majority vote
            if len(set(annot2count.values())) == 1:  # no agreement -- store label IDK
                unresolved += 1
                gt[fact] = "I Don't Know"
            else:  # agreement -- store most voted label
                gt[fact] = max(annot2count, key=annot2count.get)
    print(f'Tie breaker resolved {disagreement-unresolved} ties out of {disagreement} ({round(((disagreement-unresolved)/disagreement)*100)}%)')

    # count annotation frequency
    freq = {'Correct': 0, 'Incorrect': 0, "I Don't Know": 0}
    for annot in gt.values():
        freq[annot] += 1
    print(f'The expert-based ground truth after majority vote has the following label statistics:\n{freq}')

    #######################
    # STUDENT ANNOTATIONS #
    #######################

    print('\nStudent annotations statistics:')
    # read student files
    studFiles = glob('../data/annotations/laymen/**/annotations.csv')
    print(f'{len(studFiles)} students conducted annotations')

    # setup counter of annotations per fact and honey pot
    annotXfact = {}
    annotXpot = {}

    # set (global) annotation counters
    annotCounts = []

    # iterate over students
    for studF in studFiles:
        # read student annotations
        stud = pd.read_csv(studF, keep_default_na=False)
        # convert annotation DataFrames to dicts
        annot = {(row['Subject'], row['Predicate'], row['Object']): row['Annotation'] for ix, row in stud.iterrows()}

        # store annotation counts
        annotCounts.append(len(annot))

        # update counters per fact based on current student
        for fact in annot.keys():  # iterate over student facts
            if fact in gt:  # fact belongs to expert-based ground truth -- update honey pot counter
                if fact in annotXpot:  # honey pot fact has been already annotated by someone -- increase by 1
                    annotXpot[fact] += 1
                else:  # honey pot fact has not been annotated before -- set to 1
                    annotXpot[fact] = 1

            if fact in annotXfact:  # fact has been already annotated by someone -- increase by 1
                annotXfact[fact] += 1
            else:  # fact has not been annotated before -- set to 1
                annotXfact[fact] = 1

    # print statistics
    print(f'A total of {sum(annotCounts)} annotations have been conducted by students (laymen annotators)')
    print(f'Annotated {len(annotXfact)} distinct facts')
    print(f'Annotated {len(annotXpot)} distinct facts from honey pots')
    print(f'Average num of annotations: {round(np.mean(annotCounts))} +/- {round(np.std(annotCounts))}')
    print(f'Max num of annotations by student: {np.max(annotCounts)}')
    print(f'Min num of annotations by student: {np.min(annotCounts)}')


if __name__ == "__main__":
    main()
