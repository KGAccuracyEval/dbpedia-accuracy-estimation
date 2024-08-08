import pandas as pd

from glob import glob


def main():

    ######################
    # EXPERT ANNOTATIONS #
    ######################

    # read expert annotations
    expert1 = pd.read_csv('../data/annotations/experts/expert1/errorAnnotations.csv', keep_default_na=False)
    expert2 = pd.read_csv('../data/annotations/experts/expert2/errorAnnotations.csv', keep_default_na=False)
    expert3 = pd.read_csv('../data/annotations/experts/expert3/errorAnnotations.csv', keep_default_na=False)  # expert3 represents the tie breaker

    # convert annotation DataFrames to dicts
    annot1 = {(row['Subject'], row['Predicate'], row['Object']): row['Error'] for ix, row in expert1.iterrows()}
    annot2 = {(row['Subject'], row['Predicate'], row['Object']): row['Error'] for ix, row in expert2.iterrows()}
    annot3 = {(row['Subject'], row['Predicate'], row['Object']): row['Error'] for ix, row in expert3.iterrows()}

    # set error counter
    errorCounts = {}

    for fact, error in annot1.items():
        if error in errorCounts:
            errorCounts[error] += 1
        else:
            errorCounts[error] = 1

    for fact, error in annot2.items():
        if error in errorCounts:
            errorCounts[error] += 1
        else:
            errorCounts[error] = 1

    for fact, error in annot3.items():
        if error in errorCounts:
            errorCounts[error] += 1
        else:
            errorCounts[error] = 1

    #######################
    # STUDENT ANNOTATIONS #
    #######################

    # read student files
    studFiles = sorted(glob('../data/annotations/laymen/**/errorAnnotations.csv'))

    # iterate over students
    for studF in studFiles:
        # read student error annotations
        stud = pd.read_csv(studF, keep_default_na=False)

        # convert error DataFrames to dicts
        annot = {(row['Subject'], row['Predicate'], row['Object']): row['Error'] for ix, row in stud.iterrows()}

        # iterate over annotation data and store error annotations
        for fact, error in annot.items():
            if error in errorCounts:
                errorCounts[error] += 1
            else:
                errorCounts[error] = 1

    totCounts = sum([count for count in errorCounts.values()])

    print('Error annotation statistics')
    print(f'Total number of error annotations: {totCounts}')
    for error, count in errorCounts.items():
        print(f"{error.replace(';', '+')}: {count} ({round((count/totCounts)*100)}%)")


if __name__ == "__main__":
    main()
