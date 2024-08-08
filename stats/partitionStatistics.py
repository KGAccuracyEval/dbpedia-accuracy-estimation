import json
import pandas as pd

from glob import glob


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
    annot1 = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in expert1.iterrows()}
    annot2 = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in expert2.iterrows()}
    annot3 = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in expert3.iterrows()}

    # set dict mapping fact to its batch ID
    fact2batch = {}
    # set ground truth var
    gt = {}
    # generate expert-based ground truth
    for fact in annot1.keys():
        # set fact annotations
        if annot1[fact][0] == annot2[fact][0]:  # agreement between experts
            gt[fact] = annot1[fact][0]
        else:  # disagreement -- break ties w/ expert3
            # set annotation counter
            annot2count = {'Correct': 0, 'Incorrect': 0, "I Don't Know": 0}
            # store expert annotations
            annot2count[annot1[fact][0]] += 1
            annot2count[annot2[fact][0]] += 1
            annot2count[annot3[fact][0]] += 1

            # aggregate annotations via majority vote
            if len(set(annot2count.values())) == 1:  # no agreement -- store label IDK
                gt[fact] = "I Don't Know"
            else:  # agreement -- store most voted label
                gt[fact] = max(annot2count, key=annot2count.get)
        # map fact to its batch ID
        fact2batch[fact] = annot1[fact][1]

    # read expert metadata -- it is sufficient to use the metadata associated w/ expert1
    with open('../data/annotations/experts/expert1/metadata.json', 'r') as f:
        expertMeta = json.load(f)

    # set dict mapping fact to its stratum
    fact2stratum = {}
    # setup counter of annotations per stratum
    annotXstratum = {}
    for fact, batch in fact2batch.items():  # iterate over facts and corresponding batches
        # get stratum of current fact
        stratum = expertMeta[str(batch)]['stratum']
        # map current fact to its stratum
        fact2stratum[fact] = stratum

        if stratum not in annotXstratum:  # stratum not found before -- add stratum counters
            annotXstratum[stratum] = {'Correct': 0, 'Incorrect': 0, "I Don't Know": 0}
        # update annotation counter for the current stratum
        annotXstratum[stratum][gt[fact]] += 1

    # setup counter of strata-based annotations per fact
    countXstratum = {ix: 0 for ix in range(7)}
    for fact in gt.keys():
        countXstratum[fact2stratum[fact]] += 1

    print('\nFact assigned to experts per stratum:')
    for stratum, count in countXstratum.items():
        print(f'stratum {stratum + 1}: {count} facts')

    print('\nAnnotation statistics per stratum:')
    # sort annotXstratum
    annotXstratum = dict(sorted(annotXstratum.items()))
    # iterate over annotations per stratum and report distributions
    for stratum, annotCounts in annotXstratum.items():
        # fetch annotations counts
        correct = annotCounts["Correct"]
        incorrect = annotCounts["Incorrect"]
        idk = annotCounts["I Don't Know"]
        tot = correct + incorrect + idk
        # compute percentages
        cor_perc = round((correct / tot) * 100)
        inc_perc = round((incorrect / tot) * 100)
        idk_perc = round((idk / tot) * 100)
        # print stratum statistics
        print(f'stratum {stratum + 1}: Correct={correct} ({cor_perc}%) Incorrect={incorrect} ({inc_perc}%) IDK={idk} ({idk_perc}%)')

    #######################
    # STUDENT ANNOTATIONS #
    #######################

    print('\nStudent annotations statistics')
    # fetch student directories
    studFolders = sorted(glob('../data/annotations/laymen/**/'))

    # setup counter of strata-based annotations per fact
    countXstratum = {ix: {} for ix in range(7)}
    # setup counter of annotations per stratum
    annotXstratum = {}

    # iterate over students
    for studF in studFolders:
        # read student annotations and metadata
        stud = pd.read_csv(studF+'annotations.csv', keep_default_na=False)
        with open(studF+'metadata.json', 'r') as f:
            meta = json.load(f)
        # convert annotation DataFrames to dicts
        annot = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in stud.iterrows()}

        for fact, annot_batch in annot.items():  # iterate over annotation data
            # get batch ID
            batch = str(annot_batch[1])
            # get stratum ID
            stratum = meta[batch]['stratum']

            # update counter of annotation for current stratum
            if fact not in countXstratum[stratum]:  # fact not counted before -- add to counter
                countXstratum[stratum][fact] = 1

            if stratum not in annotXstratum:  # stratum not found before -- add stratum counters
                annotXstratum[stratum] = {'Correct': 0, 'Incorrect': 0, "I Don't Know": 0}
            # update annotation counter for the current stratum
            annotXstratum[stratum][annot_batch[0]] += 1

    print('\nFact assigned to students per stratum:')
    for stratum, facts in countXstratum.items():
        print(f'stratum {stratum + 1}: {sum(facts.values())} facts')

    print('\nAnnotation statistics per stratum:')
    # sort annotXstratum
    annotXstratum = dict(sorted(annotXstratum.items()))
    # iterate over annotations per stratum and report distributions
    for stratum, annotCounts in annotXstratum.items():
        # fetch annotations counts
        correct = annotCounts["Correct"]
        incorrect = annotCounts["Incorrect"]
        idk = annotCounts["I Don't Know"]
        tot = correct + incorrect + idk
        # compute percentages
        cor_perc = round((correct / tot) * 100)
        inc_perc = round((incorrect / tot) * 100)
        idk_perc = round((idk / tot) * 100)
        # print stratum statistics
        print(f'stratum {stratum + 1}: Correct={correct} ({cor_perc}%) Incorrect={incorrect} ({inc_perc}%) IDK={idk} ({idk_perc}%)')


if __name__ == "__main__":
    main()
