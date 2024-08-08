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

For **experts** and **laymen**, the data is divided into subfolders specified by ```{expert#|layman#}```, where # represents the annotator ID. <br>
For each annotator, we release three files:
- ```annotations.csv```, which contains the following columns:
  - **Subject:** the subject of the target fact;
  - **Predicate:** the predicate of the target fact;
  - **Object:** the object of the target fact;
  - **Annotation:** the annotation, which can take values ```{Correct, Incorrect, I Don't Know}```;
  - **BatchID:** the ID of the batch containing the target fact (i.e., the ID of the cluster sampled via TWCS);
  - **BatchTime:** the time (seconds) spent to annotate the batch containing the target fact;
  - **BatchDate:** the date the (batch) annotation task was conducted, defined as an integer representing the day within the six-week annotation period (i.e., [1, 42] range). Note that BatchDate is only available for layman annotators.
- ```errorAnnotations.csv```, which contains the same columns of ```annotations.csv``` but replaces the annotation column with:
  - **Error:** the error annotation, which can take values ```{Subject, Predicate, Object}``` or any semicolon-separated (;) combination of these three elements.
- ```metadata.json```, a dict with the following structure ```{"BatchID": {"stratum": #, "topics": ["topic1", "topic2", ...]}, ...}```.
  - **stratum**: the ID of the stratum containing the target batch;
  - **topics**: the list of topics associated with the target batch. Note that topics are only available for layman annotators.
  
  Note that topics refer to those specified by students. Hence, if a batch is associated with multiple topics, but the student specified only one topic associated with that batch, then the student's metadata only contains the specified topic and ignores the other ones. <br>
  Finally, when ```"topics": ["expert"]``` the batch represents a **honey pot**.

For **LLMs**, the annotations take the following format:
  - ```llm-name.json```, a dict with the following structure ```{"FactID": {"label": "###", "time": #.##, "retries": #}, ...}```.
    - **FactID** is an integer representing the fact ID as stored in ```./data/dataset/llm/kg.json```;
    - **label** is the LLM annotation, which can take values ```{correct, incorrect, idk, na}``` -- where ```na``` occurs when the LLM fails to provide a proper answer after four attempts;
    - **time** is the time (seconds) required by the LLM to annotate the target fact, considering all its attempts (up to four);
    - **retries** is the number of extra attempts required by the LLM to provide a proper annotation for the target fact (up to three).
    
    We release the annotations of three LLMs: ```gemma-7b.json```, ```llama3-8b.json```, and ```mistral-7b.json```.

### Datasets

We release two datasets, serving different purposes:
- ```./data/dataset/human/kg.json```: derived from the reliability-weigthed label aggregation process, consisting of 9,930 triplets and the corresponding aggregated labels in the following format ```{"StratumID": [[[subject, predicate, object], label], ...], ...}```.
- ```./data/dataset/llm/kg.json```: derived by gathering all facts annotated by expert and/or layman annotators, consisting of 11,419 triplets in the following format ```

### Sample

We also release the 

## LLM

## Estimation

## Acknowledgments
The work is partially supported by the HEREDITARY project, as part of the EU Horizon Europe research and innovation programme under Grant Agreement No GA 101137074.

## Reference
If you use or extend our work, please cite the following:

```
@inproceedings{marchesin_etal-hcomp2024,
  author = {S. Marchesin and G. Silvello and O. Alonso},
  title = {Utility-Oriented Knowledge Graph Accuracy Estimation with Limited Annotations: A Case Study on DBpedia},
  booktitle = {Proceedings of the Twelfth {AAAI} Conference on Human Computation and Crowdsourcing, {HCOMP} 2024, October 16--19, 2024, Pittsburgh, Pennsylvania, USA},
  publisher = {{AAAI} Press},
  year = {2024}
}
```
