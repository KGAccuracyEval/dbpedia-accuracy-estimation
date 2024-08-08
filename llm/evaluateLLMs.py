import json
import pandas as pd

from glob import glob
from sklearn.metrics import accuracy_score, balanced_accuracy_score


label2class = {"Correct": 2, "Incorrect": 0, "I Don't Know": 1}
llm2class = {'correct': 2, 'incorrect': 0, 'idk': 1, 'na': 1}
class2pred = {0: 'incorrect', 1: 'idk', 2: 'correct'}


def main():
    ######################
    # EXPERT ANNOTATIONS #
    ######################

    print('Expert annotators statistics')
    # read expert annotations
    expert1 = pd.read_csv('../data/annotations/experts/expert1/annotations.csv', keep_default_na=False)
    expert2 = pd.read_csv('../data/annotations/experts/expert2/annotations.csv', keep_default_na=False)
    expert3 = pd.read_csv('../data/annotations/experts/expert3/annotations.csv', keep_default_na=False)  # expert3 represents the tie breaker

    # convert annotation DataFrames to dicts
    annot1 = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in expert1.iterrows()}
    annot2 = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in expert2.iterrows()}
    annot3 = {(row['Subject'], row['Predicate'], row['Object']): [row['Annotation'], row['BatchID']] for ix, row in expert3.iterrows()}

    # read expert metadata -- it is sufficient to use the metadata associated w/ expert1
    with open('../data/annotations/experts/expert1/metadata.json', 'r') as f:
        expert_meta = json.load(f)

    expertGT = {}
    # generate expert ground truth
    for fact in annot1.keys():
        # set fact annotations
        if annot1[fact][0] == annot2[fact][0]:  # agreement between annotators
            expertGT[fact] = annot1[fact][0]
        else:  # disagreement -- break ties
            annot2class = {'Correct': 0, 'Incorrect': 0, "I Don't Know": 0}
            annot2class[annot1[fact][0]] += 1
            annot2class[annot2[fact][0]] += 1
            annot2class[annot3[fact][0]] += 1
            # aggregate annotations via majority vote
            if len(set(annot2class.values())) == 1:  # no agreement -- store label IDK
                expertGT[fact] = "I Don't Know"
            else:  # agreement -- store label
                expertGT[fact] = max(annot2class, key=annot2class.get)

    ###################
    # LLM ANNOTATIONS #
    ###################

    # read KG to get facts
    with open('../data/dataset/llm/kg.json', 'r') as f:
        kg = json.load(f)

    # read LLM files
    llmFiles = glob('../data/annotations/llms/**.json')

    gtLLMs = {}
    # iterate over LLMs
    for llmF in llmFiles:
        # read LLM annotations
        with open(llmF, 'r') as f:
            annot = json.load(f)

        # fetch LLM name
        name = llmF.split('/')[-1].split('.json')[0]
        # setup current LLM annotations
        gtLLMs[name] = {}

        for ix in kg.keys():  # iterate over fact IDs within KG
            fact = tuple(kg[ix])
            # store fact annotation by LLM and number of retries
            gtLLMs[name][fact] = [annot[ix]['label'], annot[ix]['retries']]

    for llm, annots in gtLLMs.items():  # iterate over LLM annotations and compare w/ expert ones
        # set compliance, truthfullness, and informativeness vars
        comp = []
        truth = []
        inform = []

        # set LLM (preds) and expert (trues) vars to store annotations
        preds = []
        trues = []

        for fact in expertGT.keys():  # iterate over expert-based ground truth
            humanLabel = label2class[expertGT[fact]]
            llmLabel = llm2class[annots[fact][0]]
            llmRetries = annots[fact][1]

            # compliance
            if llmRetries == 0:
                comp.append(1)
            else:
                comp.append(0)
            # truthfullness
            if llmLabel == humanLabel or llmLabel == 1:
                truth.append(1)
            else:
                truth.append(0)
            # informativeness
            if llmLabel != 1:
                inform.append(1)
            else:
                inform.append(0)
            # update preds and trues vars
            preds.append(llmLabel)
            trues.append(humanLabel)

        # compute metrics
        comp_m = round(sum(comp)/len(comp), 2)
        truth_m = round(sum(truth)/len(truth), 2)
        inform_m = round(sum(inform)/len(inform), 2)
        acc_m = round(accuracy_score(trues, preds), 2)
        bacc_m = round(balanced_accuracy_score(trues, preds), 2)
        print(f'{llm}:\nCompliance={comp_m}\nTruthfullness={truth_m}\nInformativeness={inform_m}\nAccuracy={acc_m}\nBalanced Accuracy={bacc_m}')
        print(f'Correct={sum([1 for p in preds if p == 2])} Incorrect={sum([1 for p in preds if p == 0])} IDK={sum([1 for p in preds if p == 1])}\n')


if __name__ == "__main__":
    main()
