# Experiments
### NOTE: We are refactoring this code in the `main` branch.
 
Labeling functions for weakly supervised biomedical classification tasks.

| Name             | Task             | Domain     | Type | Source                                        | Access |
|------------------|------------------|------------|------|-----------------------------------------------|------------|
| `cdr`        | Chemical, Disease | Literature | NER  | [BioCreative V Chemical-Disease Relation (CDR)](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/) | Public |
| `n2c2/i2b2`     | Drug             | Clinical   | NER  | [n2c2/i2b2 2009 Medication Challenge](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)           | DUA |
| `shareclef` | Disorder         | Clinical   | NER  | [ShARe/CLEF 2014]( https://physionet.org/content/shareclefehealth2014task2/1.0/)                               | DUA |

 
## I. Quick Start

Public datasets, dependencies, and weights are available [here](https://drive.google.com/drive/folders/1zeZPZUBlV-jh-WCDK8YnkIU3Pm-LSqwu?usp=sharing). `NER-Datasets-ALL` contains preprocessed douments, `data` contains dictionary dependencies, and `biobert_v1.1_pubmed-pytorch` contains PyTorch BioBERT weights.

See `experiments/` for scripts to run NER experiments and specific scripts for command line argument examples. For example, these scripts run the full NER pipeline for the n2c2/i2b2 `Drug` task.

- `drug_lfs.sh`
- `drug_label_model.sh`
- `drug_proba_conll.sh`


## II. Pipeline Details


This assumes your documents have been preprocessed into a JSON container format. SpaCy tools for parsing documents are found in `preprocessing/`.

### 1. Generate Label Matrices 
This applies our labeling functions to unlabeled documents. 

```
python apply-lfs.py \
--indir <INDIR> \
--outdir <OUTDIR> \
--data_root <DATA> \
--corpus cdr \
--domain pubmed \
--tag_fmt BIO \
--entity_type chemical \
--top_k 4 \
--active_tiers 1234 \
--n_procs 4 > chemical.lfs.log
```

### 2. Train Label Model
Use the default Snorkel label model to estimate per labeling function accuracies.

```
python train-label-model.py \
--indir <INDIR> \
--outdir <OUTDIR> \
--train 0 \
--dev 1 \
--test 2 \
--n_model_search 50 \
--tag_fmt_ckpnt BIO \
--seed 1234 > chemical.label_model.log
```
### 3. Generate Probabilistic Sequence Labels
Given a trained label model, create a probabalistic sequence labeled dataset for the format (where `|` is a tab delimitter). 

`TOKEN|CLASS_0_PROBA|...|CLASS_N_PROBA|UNMASKED|IOB2_TAG`

```
python label-model-proba-conll.py \
--model <LABEL_MODEL> \
--train <TRAIN_JSON>  \
--dev <DEV_JSON>   \
--test <TEST_JSON>  \
--indir <INDIR> \
--outdir <OUTDIR> \
--etype <ETYPE> > proba_sequence_labels.log
```

### 4. Train BioBERT End Model

Give the probabalsitic sequence labeled data, train a BERT model. We use BioBERT  in our experiments, but any BERT model (e.g., ClinicalBERT) will also work given the correct PyTorch weights. 

```
python proba-train-bert.py \
--train chemical.train.proba.tsv \
--dev chemical.dev.proba.tsv \
--test chemical.test.proba.tsv \
--model biobert_v1.1_pubmed-pytorch/ \
--device cuda \
--n_epochs 10 \
--lr 1e-5 \
--seed 1234 > chemical.bert.log
```
