import os
import json
import math
import pandas as pd

from glob import glob
from sklearn.metrics import cohen_kappa_score


annot2class = {"Incorrect": 0, "I Don't Know": 1, "Correct": 2}


def sigmoid(k, r=5):
    """
    Sigmoid function used to compute reliability weight

    :param k: kappa score
    :param r: rho parameter
    :return: reliability weight
    """

    return 1 / (1 + math.exp(-r*k))


def main():

    ######################
    # EXPERT ANNOTATIONS #
    ######################

    # read expert annotations
    expert1 = pd.read_csv('../data/annotations/experts/expert1/annotations.csv', keep_default_na=False)
    expert2 = pd.read_csv('../data/annotations/experts/expert2/annotations.csv', keep_default_na=False)
    expert3 = pd.read_csv('../data/annotations/experts/expert3/annotations.csv', keep_default_na=False)  # tie breaker

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
        else:  # disagreement -- break ties
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
    for fact, batch in fact2batch.items():  # iterate over facts and corresponding batches
        # get stratum of current fact
        stratum = expertMeta[str(batch)]['stratum']
        # map current fact to its stratum
        fact2stratum[fact] = stratum

    # store aggregated annotation per fact within corresponding strata
    annotXfact = {ix: {} for ix in range(7)}
    for fact in gt.keys():
        annotXfact[fact2stratum[fact]][fact] = gt[fact]

    #######################
    # STUDENT ANNOTATIONS #
    #######################

    annot2ix = {'Correct': 0, 'Incorrect': 1, "I Don't Know": 2}
    ix2annot = {0: 'Correct', 1: 'Incorrect', 2: "I Don't Know"}

    # fetch student directories
    studFolders = sorted(glob('../data/annotations/laymen/**/'))

    # set var to map student to corresponding honey pots
    stud2pot = {}

    # iterate over students
    for studF in studFolders:
        # fetch student name
        name = studF.split('/')[-2]
        # setup var to store student and expert annotations on honeypots
        honeypots = {'stud': [], 'expert': []}

        # read student annotations and metadata
        stud = pd.read_csv(studF + 'annotations.csv', keep_default_na=False)
        with open(studF + 'metadata.json', 'r') as f:
            meta = json.load(f)

        # convert annotation DataFrames to dicts
        annot = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in stud.iterrows()}

        # iterate over annotation data and store honeypot annotations
        for fact, data in annot.items():
            # get batch ID
            batch = str(data[1])

            if 'expert' in meta[batch]['topics']:  # honeypot fact -- store student and expert annotations then skip
                honeypots['stud'].append(data[0])
                honeypots['expert'].append(gt[fact])
        # store honeypot annotations for current student
        stud2pot[name] = honeypots

    # compute student weights
    stud2weight = {}
    for name, pot in stud2pot.items():
        # convert student and expert annotations into classes (i.e., indices)
        studC = [annot2class[annot] for annot in pot['stud']]
        expertC = [annot2class[annot] for annot in pot['expert']]

        # compute cohen's k over honeypots
        kappaScore = cohen_kappa_score(studC, expertC, labels=[0, 1, 2], weights='linear')
        # convert cohen's k into weight
        stud2weight[name] = sigmoid(kappaScore)

    # set var to store annotation data per fact within each stratum
    dataXfact = {ix: {} for ix in range(7)}

    # iterate over students
    for studF in studFolders:
        # fetch student name
        name = studF.split('/')[-2]

        # read student annotations and metadata
        stud = pd.read_csv(studF + 'annotations.csv', keep_default_na=False)
        with open(studF + 'metadata.json', 'r') as f:
            meta = json.load(f)

        # convert annotation DataFrames to dicts
        annot = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in stud.iterrows()}

        # iterate over annotation data and compute annotation scores based on students' weights and preferences
        for fact, data in annot.items():
            # get batch ID
            batch = str(data[1])
            # get stratum ID
            stratum = meta[batch]['stratum']

            if 'expert' in meta[batch]['topics']:  # honeypot fact -- skip it
                continue

            # get student annotation as class index
            ix = annot2ix[data[0]]

            if fact not in dataXfact[stratum]:  # new fact -- setup score and annotation counts
                dataXfact[stratum][fact] = {'classes': [0, 0, 0], 'annots': 0}
            # update score and annotation counts
            dataXfact[stratum][fact]['classes'][ix] += stud2weight[name]
            dataXfact[stratum][fact]['annots'] += 1

    # compute aggregations and store class per fact
    for stratum, factData in dataXfact.items():
        for fact, data in factData.items():
            if data['annots'] == 1:  # skip facts w/ only one annotator
                continue
            elif data['annots'] == 2:  # keep facts annotated by two annotators only when both agree
                if data['classes'].count(0) == 2:  # annotators agree on a given class
                    # get max score
                    max_score = max(data['classes'])
                    # get annotation w/ max score based on class index
                    annot = ix2annot[data['classes'].index(max_score)]
                    # store aggregated annotation
                    annotXfact[stratum][fact] = annot
                else:  # skip fact
                    continue
            else:  # three or more annotators -- standard case
                # get max score
                max_score = max(data['classes'])
                # count max score occurrences
                max_count = data['classes'].count(max_score)
                if max_count > 1:  # no agreement -- store IDK
                    annotXfact[stratum][fact] = "I Don't Know"
                else:
                    # get annotation w/ max score based on class index
                    annot = ix2annot[data['classes'].index(max_score)]
                    # store aggregated annotation
                    annotXfact[stratum][fact] = annot

    totCOR = 0
    totINC = 0
    totIDK = 0
    for stratum, annots in annotXfact.items():
        cor = 0
        inc = 0
        idk = 0
        for fact, annot in annots.items():
            if annot == 'Correct':
                cor += 1
            if annot == 'Incorrect':
                inc += 1
            if annot == "I Don't Know":
                idk += 1
        totCOR += cor
        totINC += inc
        totIDK += idk
        print(f'Stratum {stratum+1}: Correct={cor}\tIncorrect={inc}\tIDK={idk}\tTotal={cor+inc+idk}')
    print(f'KG: Correct={totCOR}\tIncorrect={totINC}\tIDK={totIDK}\tTotal={totCOR+totINC+totIDK}')

    # convert derived data into json format and store
    kg = {stratum: [(fact, annot) for fact, annot in facts.items()] for stratum, facts in annotXfact.items()}

    # store dataset containing aggregated label obtained from three annotations or two consistent ones
    os.makedirs('../data/dataset/human/', exist_ok=True)
    with open('../data/dataset/human/kg.json', 'w') as out:
        json.dump(kg, out)


if __name__ == "__main__":
    main()
