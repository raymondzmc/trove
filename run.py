import sys
import os
import re
import pdb
import argparse
from os.path import join as pjoin
from pathlib import Path
import json
import itertools
from tqdm import tqdm
from scipy.sparse import dok_matrix, vstack, csr_matrix
from snorkel.labeling.model.label_model import LabelModel

import torch
import scispacy
import spacy
import pandas as pd
import numpy as np
import transformers
from collections import defaultdict
from preprocessing.pipes.tokenizers import parse_doc
from trove.labelers.umls import UMLS, SemanticGroups
from trove.transforms import SmartLowercase
from trove.dataloaders import load_json_dataset
from trove.labelers.labeling import OntologyLabelingFunction
from trove.labelers.abbreviations import SchwartzHearstLabelingFunction
from trove.labelers.core import SequenceLabelingServer
from trove.metrics.analysis import lf_summary
from trove.models.model_search import grid_search
from trove.utils import score_umls_ontologies, combine_ontologies


ROOT_DIR = os.getcwd()


def install_from_zip(args):

    # setup defaults
    UMLS.config(
        cache_root = pjoin(ROOT_DIR, '.cache'),
        backend = 'pandas'
    )


    if not UMLS.is_initalized():
        print("Initializing the UMLS from zip file...")
        UMLS.init_from_nlm_zip(args.zipfile_path)

def test_installation(args):


    # english stopwords
    # stopwords = set(open('data/stopwords.txt','r').read().splitlines())
    # stopwords = stopwords.union(set([t[0].upper() + t[1:] for t in stopwords]))

    # options for filtering terms
    config = {
        "type_mapping"  : "TUI",  # TUI = semantic types, CUI = concept ids
        'min_char_len'  : 2,
        'max_tok_len'   : 8,
        'min_dict_size' : 500,
        'stopwords'     : set(),
        'transforms'    : [SmartLowercase()],
        'languages'     : {"ENG"},
        'filter_sabs'   : {"SNOMEDCT_VET"},
        'filter_rgx'    : r'''^[-+]*[0-9]+([.][0-9]+)*$'''  # filter numbers
    }

    umls = UMLS(**config)
    pdb.set_trace()
    # For testing
    semgroups = SemanticGroups()
    stys = umls.terminologies['MSH']['acetaminophen']
    print(stys)
    print([semgroups.types[sty] for sty in stys])


def preprocess_msgs(args):
    nlp = spacy.load("en_core_sci_sm")
    df = pd.read_csv(args.csvfile_path, sep=',')
    out_path = Path(args.jsonfile_path)

    pbar = tqdm(total=len(df))

    with out_path.open("w", encoding="utf8") as f:
        for index, row in df.iterrows():
            sents = list(parse_doc(nlp(row['text']), disable=['lemmatizer', 'ner']))
            metadata = {'encounter_id': row['encounter_id'],
                        'occurs_at': row['occurs_at'],
                        'role': row['role'],
                        'is_followup': row['is_followup']}
            f.write(
                json.dumps(
                    {'name': str(index),
                     'metadata': metadata,
                     'sentences': sents}
                )
            )
            f.write("\n")
            pbar.update(1)

    print("Saved {} texts to JSON {}".format(len(df), out_path))
    pdb.set_trace()

def preprocess_msgs_kb_responses(args):
    nlp = spacy.load("en_core_sci_sm")
    df = pd.read_csv(args.csvfile_path, sep=',')
    out_path = Path(args.jsonfile_path)

    pbar = tqdm(total=len(df))

    with out_path.open("w", encoding="utf8") as f:
        for index, row in df.iterrows():
            sents = list(parse_doc(nlp(row['text'].split('|')[1]), disable=['lemmatizer', 'ner']))
            metadata = {'encounter_id': row['mother_encounter_id'],
                        'occurs_at': row['occurs_at'],
                        # 'role': row['role'],
                        'response': row['text'].split('|')[0]}
            f.write(
                json.dumps(
                    {'name': str(index),
                     'metadata': metadata,
                     'sentences': sents}
                )
            )
            f.write("\n")
            pbar.update(1)

    print("Saved {} texts to JSON {}".format(len(df), out_path))

def map_entity_classes(dictionary, class_map):
    """
    Given a dictionary, create the term entity class probabilities.
    The probability of a term is determined by the labels of all its semantic types (include or not)
    """
    k = len([y for y in set(class_map.values()) if y != -1])
    ontology = {}
    for term in dictionary:
        proba = np.zeros(shape=k).astype(np.float32)
        for cls in dictionary[term]:
            # ignore abstains
            idx = class_map[cls] if cls in class_map else -1
            if idx != -1:
                proba[idx - 1] += 1
        # don't include terms that don't map to any classes
        if np.sum(proba) > 0:
            ontology[term] = proba / np.sum(proba)
    return ontology




def create_lfs(args):
    stopwords = set(open('data/stopwords.txt','r').read().splitlines())
    stopwords = stopwords.union(set([t[0].upper() + t[1:] for t in stopwords]))

    # options for filtering terms
    config = {
        "type_mapping"  : "TUI",  # TUI = semantic types, CUI = concept ids
        'min_char_len'  : 2,
        'max_tok_len'   : 8,
        'min_dict_size' : 500,
        'stopwords'     : stopwords,
        'transforms'    : [SmartLowercase()],
        'languages'     : {"ENG"},
        'filter_sabs'   : {"SNOMEDCT_VET"},
        'filter_rgx'    : r'''^[-+]*[0-9]+([.][0-9]+)*$'''  # filter numbers
    }

    cache_root = pjoin(ROOT_DIR, '.cache')
    # setup defaults
    UMLS.config(
        cache_root = cache_root,
        backend = 'pandas'
    )


    if not UMLS.is_initalized():
        print("Initializing the UMLS from zip file...")
        UMLS.init_from_nlm_zip(args.zipfile_path)

    # sty_df = pd.read_csv(pjoin(cache_root, 'SemGroups.txt'), sep="|", names=['GRP', 'GRP_STR', 'TUI', 'STR'])
    sty_df = pd.read_csv(pjoin('data', 'Trove Semantic Types - Sheet1.csv'))
    class_map = {}
    for row in sty_df.itertuples():
        semantic_type = row[1]
        tui = semantic_type.split('|')[2]
        label = int(row[-1]) if (not np.isnan(row[-1])) else 0
        class_map[tui] = label
        # class_map[row.TUI] = 1


    umls = UMLS(**config)

    use_top_s = False
    s = 4 # This hyperparameter will need to be fine-tuned on a dev/val set

    if use_top_s:

        ontology_scores_file = pjoin(cache_root, 'data_sources.npy')
        if not os.path.isfile(ontology_scores_file):
            all_sources = list(umls.terminologies.keys())
            tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            dataset = load_json_dataset(pjoin(ROOT_DIR, 'data', 'kb_responses.json'), tokenizer)
            sentences = dataset.sentences
            scores = score_umls_ontologies(sentences, umls.terminologies)
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            np.save(ontology_scores_file, sorted_scores)
        else:
            sorted_scores = np.load(ontology_scores_file)

        to_keep = [x[0] for x in sorted_scores[:s]]
        to_combine = [x[0] for x in sorted_scores[s:]]


        # Move the rest of data sources into a single ontology
        umls.terminologies['others'] = combine_ontologies(umls.terminologies, to_combine)
        terminologies = to_keep + ['others']

    else:
        terminologies = ['CHV', 'SNOMEDCT_US', 'RXNORM']

    ontologies = {
        sab : map_entity_classes(umls.terminologies[sab], class_map)
        for sab in terminologies
    }

    
    # create dictionaries for our Schwartz-Hearst abbreviation detection labelers
    positive, negative = set(), set()

    for sab in umls.terminologies:
        for term in umls.terminologies[sab]:
            for tui in umls.terminologies[sab][term]:
                if tui in class_map and class_map[tui] == 1:
                    positive.add(term)
                elif tui in class_map and class_map[tui] == 0:
                    negative.add(term)

    stopwords = {t:2 for t in stopwords}

    ontology_lfs = [
        OntologyLabelingFunction(
            f'UMLS_{name}', 
            ontologies[name], 
            stopwords=stopwords 
        )
        for name in ontologies
    ]
    # ontology_lfs += [
    #     SchwartzHearstLabelingFunction('UMLS_schwartz_hearst_1', positive, 1, stopwords=stopwords),
    #     SchwartzHearstLabelingFunction('UMLS_schwartz_hearst_2', negative, 2)
    # ]
    return ontology_lfs

def create_word_lf_mat(Xs, Ls, num_lfs):
    """
    Create word-level LF matrix from LFs indexed by sentence/word
    0 words X lfs
    1 words X lfs
    2 words X lfs
    ...
    
    """
    Yws = []
    for sent_i in range(len(Xs)):
        ys = dok_matrix((len(Xs[sent_i].words), num_lfs))
        for lf_i in range(num_lfs):
            for word_i,y in Ls[sent_i][lf_i].items():
                ys[word_i, lf_i] = y
        Yws.append(ys)
    return csr_matrix(vstack(Yws))

def convert_label_matrix(L):
    # abstain is -1
    # negative is 0
    L = L.toarray().copy()
    L[L == 0] = -1
    L[L == 2] = 0
    return L

# def visualize_evaluation(X_sents, ):


def construct_label_matrix(args, lfs):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    dataset = load_json_dataset(pjoin(ROOT_DIR, 'data', 'kb_responses.json'), tokenizer)

    X_sents = [dataset.sentences]
    labeler = SequenceLabelingServer(num_workers=4)
    L_sents = labeler.apply(lfs, X_sents)


    X_words = np.array(list(itertools.chain.from_iterable([s.words for s in X_sents[0]])))
    X_seq_lens = np.array([len(s.words) for s in X_sents[0]])

    X_doc_seq_lens = np.array([len(doc.sentences) for doc in dataset.documents])

    Y_words = [dataset.tagged(i)[-1] for i in range(len(dataset))]
    # pdb.set_trace()Chemical
    # tag2idx = {'O':2, 'I-':1}
    # No GT Label
    Y_words = np.array([2 for t in list(itertools.chain.from_iterable(Y_words))])
    L_words = create_word_lf_mat(X_sents[0], L_sents[0], len(lfs))
    # lf_summary(L_words, Y=Y_words, lf_names=[lf.name for lf in lfs])

    L_words_hat = convert_label_matrix(L_words)
    Y_words_hat = np.array([0 if y == 2 else 1 for y in Y_words])
    np.random.seed(1234)

    n = L_words_hat.shape[0]

    param_grid = {
        'lr': [0.01, 0.005, 0.001, 0.0001],
        'l2': [0.001, 0.0001],
        'n_epochs': [50, 100, 200, 600, 700, 1000],
        'prec_init': [0.6, 0.7, 0.8, 0.9],
        'optimizer': ["adamax"], 
        'lr_scheduler': ['constant'],
    }

    model_class_init = {
        'cardinality': 2, 
        'verbose': True
    }

    n_model_search = 25
    # num_hyperparams = functools.reduce(lambda x,y:x*y, [len(x) for x in param_grid.values()])
    # print("Hyperparamater Search Space:", num_hyperparams)

    model = LabelModel(**model_class_init)
    L_train= L_words_hat
    model.fit(L_train)
    y_pred = model.predict(L_train)
    # pdb.set_trace()

    X_sents= X_sents[0]
    words = []
    prev_end_index = 0

    mv_extracted = 0
    for i, sent in enumerate(X_sents):
        y_pred_ = y_pred[prev_end_index: prev_end_index + len(sent.words)]
        L_train_ = L_train[prev_end_index: prev_end_index + len(sent.words)]

        y_pred_mv = np.zeros(len(sent.words))
        for i, votes in enumerate(L_train_):
            values, count = np.unique(votes, return_counts=True)
            max_count = count.max()

            if len(count[count == max_count]) > 1:
                mv = -1
            else:
                mv = values[count.argmax()]
            y_pred_mv[i] = mv

        mv_extracted += len(y_pred_mv[y_pred_mv == 1])

        extracted_words_lfs = np.array(sent.words)[y_pred_ == 1]
        extracted_words_mv = np.array(sent.words)[y_pred_mv == 1]
        prev_end_index += len(sent.words)
        words.append([sent.words, extracted_words_lfs, extracted_words_mv]) 

    print(f"MV coverage: {mv_extracted / len(y_pred)}")
    print(f"LM coverage: {len(y_pred[y_pred == 1]) / len(y_pred)}")
    df = pd.DataFrame(words)
    df[0] = df[0].map(lambda x: ' '.join(x))
    # df.to_csv('data/kb_responses_snorkle_results_v4.csv')
    

    # label_model, best_config = grid_search(LabelModel, 
    #                                        model_class_init, 
    #                                        param_grid,
    #                                        train = (L_train, Y_train, X_seq_lens[0]),
    #                                        dev = (L_dev, Y_dev, X_seq_lens[1]),
    #                                        n_model_search=n_model_search, 
    #                                        val_metric='f1', 
    #                                        seq_eval=True,
    #                                        seed=1234,
    #                                        tag_fmt_ckpnt='IO')

# def train_lf
    # X_sents = [
    #     dataset['train'].sentences,
    #     dataset['dev'].sentences,
    #     dataset['test'].sentences,
    # ]

    # labeler = SequenceLabelingServer(num_workers=4)
    # L_sents = labeler.apply(lfs, X_sents)
# Problem: 1. val set to select hyperparam 2. Requires task-specific LF, no compare snorkle vs mv (ontology only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-zipfile_path', default=pjoin(ROOT_DIR, 'umls-2020AB-metathesaurus.zip'))
    parser.add_argument('-csvfile_path', default=pjoin(ROOT_DIR, 'data', 'bq-results-20210921-134216-neshn2bak3zx.csv'))
    parser.add_argument('-jsonfile_path', default=pjoin(ROOT_DIR, 'data', 'all_msgs.json'))
    args = parser.parse_args()

    # install_from_zip(args)
    # test_installation(args)
    # preprocess_msgs(args)
    # preprocess_msgs_kb_responses(args)
    lfs = create_lfs(args)
    construct_label_matrix(args, lfs)



