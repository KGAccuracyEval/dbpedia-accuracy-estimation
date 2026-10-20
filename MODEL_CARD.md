# dbpedia-accuracy-estimation

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

| Field | Value |
|---|---|
| Name | dbpedia-accuracy-estimation |
| Version | No tagged version (main branch) |
| Type | Software or library; Dataset |
| Owner partner and contact | University of Padua (UNIPD), Stefano Marchesin, stefano.marchesin@unipd.it |
| License | MIT (see LICENSE file) |
| Repository and DOI | https://github.com/KGAccuracyEval/dbpedia-accuracy-estimation (no Zenodo DOI) |
| Release date | 8 August 2024 |
| Related task or deliverable | T3.2 |
| How to cite | S. Marchesin, G. Silvello and O. Alonso. Utility-Oriented Knowledge Graph Accuracy Estimation with Limited Annotations: A Case Study on DBpedia. Proc. AAAI HCOMP 12(1): 105-114, 2024. doi:10.1609/hcomp.v12i1.31605 |

## 2. Intended use

**Primary intended use.** Code and data to estimate the accuracy of DBpedia 2015-10 with a utility-oriented framework. Facts are weighted by their use in SPARQL query logs, the KG is stratified by utility, batches of facts are sampled with TWCS within each stratum, and human annotations are aggregated by reliability-weighted majority voting. The release also contains the annotations of three open LLMs, to study whether LLMs can replace human annotators.

**Intended users.** Researchers working on KG quality, crowdsourcing and LLM-based evaluation.

**Out-of-scope uses.** This release must not be used for
- clinical diagnosis, prognosis or treatment of individual patients;
- decisions about individuals, such as insurance, employment or access to care;
- re-identification of individuals or linkage with other data sources;
- populations, devices or data types not described in Section 3;
- use of the LLM annotations as ground truth: the LLMs overestimate DBpedia accuracy (see Section 6).

## 3. Data provenance

| Dataset | Source partner | Type (public, synthetic, pseudonymized, anonymized) | Population (size, age, sex, country, condition) | Ethics approval or access conditions |
|---|---|---|---|---|
| DBpedia 2015-10 (English): 9,930 triples with aggregated labels (`data/dataset/human`) and 11,419 annotated triples (`data/dataset/llm`) | External: Auer et al., ISWC 2007 | Public | Not applicable (KG triples; no human population) | Publicly available |
| LSQ 2.0 SPARQL query logs (utility model) | External: Stadler et al., Semantic Web 15(1), 2024 | Public | Not applicable | Publicly available; used to compute utility scores, not redistributed |
| Human correctness annotations | UNIPD, collected for this study | Anonymized | 3 expert and 60 student annotators; annotators appear only as sequential IDs; no information about the annotators is retained | Released in anonymized form |
| LLM annotations (Gemma 7B, Llama 3 8B, Mistral 7B) | UNIPD, generated for this study | Synthetic (model-generated labels) | Not applicable | None |

This repository and the released files contain no personal data, pseudonymized records, identifiers or pseudonymization keys.

## 6. Known limitations and risks

- **Generalizability.** Labels refer to the DBpedia 2015-10 snapshot: annotators were asked to assume the year 2015, so a label may not hold for later DBpedia versions. The utility model reflects the query logs of LSQ 2.0.
- **Known biases and confounders.** Agreement between the two experts on the honey pots was moderate (Cohen's kappa 0.51). Student annotators came from a single master's course and declared their own topic expertise. The LLMs rarely labeled a triple as incorrect, which inflates LLM-based accuracy estimates.
- **Failure modes.** Low-utility strata received more "I don't know" labels and required longer annotation times. Triples without at least three annotations or two consistent ones were discarded before estimation. LLM balanced accuracy against expert labels did not exceed 0.46.
- **Privacy risks.** No trained model. Human annotations are released in anonymized form.

## 7. Responsible use and reporting

- **Terms of use.** MIT license. Research use only, as stated in the disclaimer above.
- **Contact for misuse or vulnerability reports.**

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.
