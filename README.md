# Utility-Oriented Knowledge Graph Accuracy Estimation with Limited Annotations: A Case Study on DBpedia
Knowledge Graphs (KGs) are essential for applications like search, recommendation, and virtual assistants, where their accuracy directly impacts effectiveness. However, due to their large-scale and ever-evolving nature, it is impractical to manually evaluate all KG contents. We propose a framework that employs sampling, estimation, and active learning to audit KG accuracy in a cost-effective manner. The framework prioritizes KG facts based on their utility to downstream tasks. We applied the framework to DBpedia and gathered annotations from both expert and layman annotators. We also explored the potential of Large Language Models (LLMs) as KG evaluators, showing that while they can perform comparably to low-quality human annotators, they tend to overestimate KG accuracy. As such, LLMs are currently insufficient to replace human crowdworkers in the evaluation process. The results also provide insights into the scalability of methods for auditing KGs.

## Contents

This repository contains the source code and data used to estimate DBpedia accuracy. <br>
Instructions on installation, acquisition and preparation of the data used for the experiments, as well as statistics and analyses are reported below.

## Installation 

Clone this repository

```bash
git clone https://github.com/KGAccuracyEval/dbpedia-accuracy-estimation.git
```

Install Python 3.10 (preferably in a virtual environment). <br>
Install all the requirements:

```bash
pip install -r requirements.txt
```

## Data

Below, we describe the collected data, which we release in anonymized format.

### Annotations

The annotations conducted by expert and layman annotators, as well as those by LLMs can be found in:
- ```./data/annotations/experts/``` for experts;
- ```./data/annotations/laymen/``` for students;
- ```./data/annotations/llms/``` for LLMs

For experts and laymen, the data is divided into subfolders specified by ```{expert#|layman#}```, where # represents the annotator ID. <br>
For each annotator, we release three files:
- ```annotations.csv```, which contains the following columns:
  - **Subject:** the subject of the target fact;
  - **Predicate:** the predicate of the target fact;
  - **Object:** the object of the target fact;
  - **Annotation:** the annotation, which can take values ```{Correct, Incorrect, I Don't Know}```;
  - **BatchID:** the ID of the batch containing the target fact (i.e., the ID of the cluster sampled via TWCS);
  - **BatchTime:** the time spent to annotate the batch containing the target fact;
  - **BatchDate:** the date the (batch) annotation task was conducted, defined as an integer representing the day within the six-week annotation period (i.e., [1, 42] range). Note that BatchDate is only available for layman annotators.
- ```errorAnnotations.csv```, which contains the same columns of ```annotations.csv``` but replaces the annotation column with:
  - **Error:** the error annotation, which can take values ```{Subject, Predicate, Object}``` or any semicolon-separated (;) combination of these three elements.
- ```metadata.json```, a dict with the following structure ```{"BatchID": {"stratum": #, "topics": ["topic1", "topic2", ...]}, ...}```.
  - **stratum**: the ID of the stratum containing the target batch;
  - **topics**: the list of topics associated with the target batch. Note that topics are only available for layman annotators. Also note that topics refer to those specified by students. Hence, if a batch is associated with multiple topics, but the student specified only one topic associated with that batch, then the student's metadata only contains the specified topic and ignores the other ones. Finally, when ```"topics": ["expert"]``` the batch represents a **honey pot**.

## LLM

## Estimation
