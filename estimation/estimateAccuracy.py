import json
import pandas as pd
import estimationStrategies

from glob import glob


label2class = {'Correct': 1, 'Incorrect': 0}


def main():

    ######################
    # DATASET PROCESSING #
    ######################

    # read dataset
    with open('../data/dataset/human/kg.json', 'r') as f:
        kg = json.load(f)

    # prepare dataset for estimation process
    kg = {int(stratum): {tuple(fact_annot[0]): fact_annot[1] for fact_annot in facts} for stratum, facts in kg.items()}

    # setup var to store clustered facts
    clustFacts = {stratum: [] for stratum in range(7)}
    # setup var to store current cluster ID within each stratum
    current2id = {stratum: -1 for stratum in range(7)}
    # setup var to store current cluster index within each stratum -- var used to access and update current cluster
    current2ix = {stratum: -1 for stratum in range(7)}

    # setup var to store observed facts
    observed = {}

    # fetch student directories
    studFolders = sorted(glob('../data/annotations/laymen/**/'))

    # iterate over students
    for studF in studFolders:
        # read student annotations and metadata
        stud = pd.read_csv(studF + 'annotations.csv', keep_default_na=False)
        with open(studF + 'metadata.json', 'r') as f:
            meta = json.load(f)

        # convert annotation DataFrames to dicts
        annot = {(row['Subject'], row['Predicate'], row['Object']): row['BatchID'] for ix, row in stud.iterrows()}

        # iterate over facts and cluster them
        for fact, batch in annot.items():
            if fact in observed:  # fact has been already observed -- skip
                continue
            # store fact to observed
            observed[fact] = 1

            # get stratum ID
            stratum = meta[str(batch)]['stratum']
            if fact not in kg[stratum]:  # fact not in dataset -- skip
                continue

            # get label
            label = kg[stratum][fact]

            if batch == current2id[stratum]:  # batch ID matches current one -- append label to current cluster
                if label != "I Don't Know":  # label must be != IDK
                    clustFacts[stratum][current2ix[stratum]].append(label2class[label])
            else:  # batch ID does not match current one -- setup new cluster and update current batch ID and index
                if label != "I Don't Know":  # label must be != IDK
                    clustFacts[stratum].append([label2class[label]])
                    current2id[stratum] = batch  # update batch ID
                    current2ix[stratum] += 1  # update index

    # read expert annotations -- it is sufficient to use the metadata associated w/ expert1
    expert = pd.read_csv('../data/annotations/experts/expert1/annotations.csv', keep_default_na=False)
    # convert annotation DataFrames to dicts
    annot = {(row['Subject'], row['Predicate'], row['Object']): row['BatchID'] for ix, row in expert.iterrows()}
    # read expert metadata -- it is sufficient to use the metadata associated w/ expert1
    with open('../data/annotations/experts/expert1/metadata.json', 'r') as f:
        meta = json.load(f)

    # iterate over facts and cluster them
    for fact, batch in annot.items():
        if fact in observed:  # fact has been already stored -- skip
            continue
        # store fact to observed
        observed[fact] = 1

        # get stratum ID
        stratum = meta[str(batch)]['stratum']
        if fact not in kg[stratum]:  # fact not in dataset -- skip
            continue

        # get label
        label = kg[stratum][fact]

        if batch == current2id[stratum]:  # batch ID matches current one -- append label to current cluster
            if label != "I Don't Know":  # label must be != IDK
                clustFacts[stratum][current2ix[stratum]].append(label2class[label])
        else:  # batch ID does not match current one -- setup new cluster and update current batch ID and index
            if label != "I Don't Know":  # label must be != IDK
                clustFacts[stratum].append([label2class[label]])
                current2id[stratum] = batch  # update batch ID
                current2ix[stratum] += 1  # update index

    ######################
    # ACCURACY ESTIMATES #
    ######################

    # set partition estimator
    twcs = estimationStrategies.TWCSEstimator()
    # set graph estimator
    ss = estimationStrategies.STWCSEstimator()

    # load strata weights
    strataWeights = pd.read_csv('../data/sample/weights.csv', keep_default_na=False)
    strataWeights = strataWeights.loc[0, :].tolist()

    # set vars to store accuracy and accuracy variances
    accs = []
    accVars = []

    # compute partition accuracy estimates
    for stratumID, stratumClusters in clustFacts.items():
        # compute accuracy estimate and corresponding MoE
        acc = twcs.estimate(stratumClusters)
        var = twcs.computeVar(stratumClusters)
        moe = twcs.computeMoE(var)
        # store computed estimates
        accs.append(acc)
        accVars.append(var)

        print(f'Stratum {stratumID+1}: {round(acc, 2)} +/- {round(moe, 2)}')

    # compute kg accuracy estimate
    acc = ss.estimate(accs, strataWeights)
    var = ss.computeVar(accVars, strataWeights)
    moe = ss.computeMoE(var)
    print(f'KG: {round(acc, 2)} +/- {round(moe, 2)}')


if __name__ == "__main__":
    main()
