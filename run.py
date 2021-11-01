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
from nltk.corpus import stopwords

import torch
import pickle
import scispacy
import spacy
import pandas as pd
import numpy as np
import transformers
import itertools
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
from trove.dataloaders.contexts import Document, Sentence, Annotation

from curai.kb.utils import load_curai_kb
from curai.kb.enums import SnomedAttrType, SourceType

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
    concepts = []
    with out_path.open("w", encoding="utf8") as f:
        for index, row in df.iterrows():
            sents = list(parse_doc(nlp(row['text'].split('|')[1]), disable=['lemmatizer', 'ner']))
            if len(sents) == 0:
                sents = list(parse_doc(nlp('n/a'), disable=['lemmatizer', 'ner']))
            elif len(sents) > 1:
                for i in range(1, len(sents)):
                    sents[0]['words'].extend(sents[i]['words'])
                    sents[0]['abs_char_offsets'].extend(sents[i]['abs_char_offsets'])
                    sents[0]['pos_tags'].extend(sents[i]['pos_tags'])
                    sents[0]['dep_parents'].extend(sents[i]['dep_parents'])
                    sents[0]['dep_labels'].extend(sents[i]['dep_labels'])
            metadata = {
                'question': row['question_text'].split('|')[1],
                'question_concept': row['question_text'].split('|')[0],
            }
            concepts.append([metadata['question'], metadata['question_concept']])
            f.write(
                json.dumps({
                    'name': str(index),
                    'metadata': metadata,
                    'sentences': [sents[0]],
                    }
                )
            )
            f.write("\n")
            pbar.update(1)

    df = pd.DataFrame(concepts)
    df.to_csv('data/question_concepts.csv')

    print("Saved {} texts to JSON {}".format(len(df), out_path))


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


    cached_terminologies_path = pjoin(cache_root, 'terminologies.pickle')
    if not os.path.exists(cached_terminologies_path):

        UMLS.config(
            cache_root = cache_root,
            backend = 'pandas'
        )

        if not UMLS.is_initalized():
            print("Initializing the UMLS from zip file...")
            UMLS.init_from_nlm_zip(args.zipfile_path)

        umls = UMLS(**config)
        umls_terminologies = umls.terminologies
        del umls

        with open(cached_terminologies_path, 'wb') as f:
            pickle.dump(umls_terminologies, f, pickle.HIGHEST_PROTOCOL)

    # Loaded cached "terminologies" file
    else:
        with open(cached_terminologies_path, "rb") as f:
            umls_terminologies = pickle.load(f)

    # sty_df = pd.read_csv(pjoin(cache_root, 'SemGroups.txt'), sep="|", names=['GRP', 'GRP_STR', 'TUI', 'STR'])
    sty_df = pd.read_csv(pjoin('data', 'Trove Semantic Types - Sheet1.csv'))
    class_map = {}
    tuis_to_keep = set()
    for row in sty_df.itertuples():
        semantic_type = row[1]
        tui = semantic_type.split('|')[2]
        label = int(row[-1]) if (not np.isnan(row[-1])) else row[-2]
        class_map[tui] = label

        if label == 1:
            tuis_to_keep.add(tui)
        # class_map[row.TUI] = 1


    use_top_s = True
    s = 0 # This hyperparameter will need to be fine-tuned on a dev/val set

    if use_top_s:

        ontology_scores_file = pjoin(cache_root, 'data_sources.npy')
        if not os.path.isfile(ontology_scores_file):
            all_sources = list(umls_terminologies.keys())
            tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            dataset = load_json_dataset(pjoin(ROOT_DIR, 'data', 'kb_responses.json'), tokenizer)
            sentences = dataset.sentences
            scores = score_umls_ontologies(sentences, umls_terminologies)
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            np.save(ontology_scores_file, sorted_scores)
        else:
            sorted_scores = np.load(ontology_scores_file)

        to_keep = [x[0] for x in sorted_scores[:s]]
        to_combine = [x[0] for x in sorted_scores[s:]]


        # Move the rest of data sources into a single ontology
        umls_terminologies['others'] = combine_ontologies(umls_terminologies, to_combine)
        terminologies = to_keep + ['others']

    else:
        terminologies = ['CHV', 'SNOMEDCT_US', 'RXNORM']

    # ontologies = {
    #     sab : map_entity_classes(umls_terminologies[sab], class_map)
    #     for sab in terminologies
    # }

    ontologies = {
        'others' : map_entity_classes(umls_terminologies['others'], class_map)
    }

    # for term, tuis in tqdm(umls_terminologies['others'].items()):
    #     if len(set(tuis) - tuis_to_keep) == 0:
    #         del ontologies['others'][term]


    
    # create dictionaries for our Schwartz-Hearst abbreviation detection labelers
    # positive, negative = set(), set()

    # for sab in umls.terminologies:
    #     for term in umls.terminologies[sab]:
    #         for tui in umls.terminologies[sab][term]:
    #             if tui in class_map and class_map[tui] == 1:
    #                 positive.add(term)
    #             elif tui in class_map and class_map[tui] == 0:
    #                 negative.add(term)

    stopwords = {t:2 for t in stopwords}
    stopwords = {}

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
    print("Finished Initializing the Ontology-based Labeling Functions!")
    return ontology_lfs, terminologies, umls_terminologies

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




def get_entity_candidates(args, lfs, terminologies, umls_terminologies):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    dataset = load_json_dataset(args.jsonfile_path, tokenizer)
    rows = []

    curai_kb = load_curai_kb(load_dxplain=False, load_labs=False)
    item_dict = curai_kb.get_item_dict([SnomedAttrType.FINDING])
    tuis_to_pair = {'T020','T190','T049','T019','T047','T050','T033','T037','T048','T191','T046','T184'}

    unfound_q_concepts = 0
    pbar = tqdm(total=len(dataset.sentences))
    candidate_len = []

    to_keep = set(['right', 'today', 'white', 'week', 'pee', 'eat', 'left side', 'sleep', 'mucus', 'period', 
    'at night', 'right side', 'left', 'water', 'weeks', 'urinating', 'medication', 'walk', 'tylenol', 
    '4 days', 'ibuprofen', 'swallow', '3 days', 'drinking', 'periods', 'intercourse', 'no pain', 
    'both sides', '5 days', 'chlamydia', 'phlegm', 'birth control', 'movement', 'vape', 'taste', 
    'pill', 'advil', 'meds', 'albuterol', 'sudden', 'antibiotics', 'standing', 'pills', 'nights', 
    'bowel movement', 'attention', 'body aches', 'scratching', 'overweight', 'exercise', 'amoxicillin',
    'iron', 'black', 'kids', 'one side', 'feel pain', 'nexplanon', 'baby', 'loose stools', 'omeprazole',
    'opening', 'inflamed', 'chronic', 'night time', 'drinks', 'leave', 'crying', 'pressure', 'drank', 
    'children', 'diarrhea', 'spit', 'mornings', 'menstrual cycle', 'metformin', 'unprotected sex',
    'anxiety', 'benadryl', 'azo', 'mucinex', 'eating', 'localized', 'itchy', 'sharp pain', 'afraid',
    'straight', 'zyrtec', 'sharp pains', 'oxygen', 'coughing'])

    remove_singleton = set()
    with open('non_symptom_candidates.txt', 'r') as f:
        for line in f:
            candidate, count = line.split(':')[0], int(line.split(':')[1])
            if count > 5 and candidate not in to_keep:
                remove_singleton.add(candidate)


    stop_words = stopwords.words('english')

    connected_candidates = set()

    non_symptom_candidates = defaultdict(int)
    for i, sentence in enumerate(dataset.sentences):
        row = {}

        question_concept_id = sentence.document.metadata['question_concept'].split('#')[-1]

        # Ignoring those concepts not in the findings
        if question_concept_id not in item_dict.keys():
            unfound_q_concepts += 1
            # TO DO: Why
            continue
        else:
            question_concept = item_dict[question_concept_id].name

        row['question_text'] = sentence.document.metadata['question']
        row['question_concept'] = (question_concept_id, question_concept)
        row['response_text'] = ' '.join(sentence.words)
        # row['connected_candidates'] =

        lf_results = lfs[0](sentence, True)[1]
        all_matches = set(list(map(lambda x: x[1], lf_results)))
        all_matches = sorted(all_matches, key=lambda x: len(x), reverse=True)

        # Remove "my" from the words
        sentence.words = list(filter(lambda word: word.lower() != 'my', sentence.words))

        candidate_indices = []
        matched_index = np.zeros(len(sentence.words))
        for matched in all_matches:
            token_len = len(matched.split(' '))
            for word_index in range(len(sentence.words)):
                if ' '.join(sentence.words[word_index: word_index + token_len]) == matched:
                    candidate_indices.append((word_index, word_index + token_len, matched))
                    matched_index[word_index: word_index + token_len] = 1

        for word_index in range(len(matched_index)):
            word_label = matched_index[word_index]
            if word_label == 0 and sentence.words[word_index] in stop_words:
                matched_index[word_index] = 2

        consequent_candidates = []
        start = 0
        current_candidate = [sentence.words[0]] if matched_index[0] == 1 else []
        stop_word_flag = False
        for i in range(1, len(sentence.words)):
            if matched_index[i] == 1:
                current_candidate.append(sentence.words[i])

                if (i + 1 == len(sentence.words)) and len(current_candidate) > 1:
                    consequent_candidates.append(' '.join(current_candidate))

            elif matched_index[i] == 2 and (i + 1 < len(sentence.words)) and matched_index[i + 1] == 1 and len(current_candidate) > 1:
                current_candidate.append(sentence.words[i])
            else:
                if len(current_candidate) > 1:
                    consequent_candidates.append(' '.join(current_candidate))
                    print(current_candidate)
                current_candidate = []

        consequent_candidates = list(set(consequent_candidates) - set(all_matches))

        # stopword_separated_candidates = []
        # for (cand1, cand2) in itertools.combinations(candidate_indices, 2):

        #     # if cand1[1] == cand2[0]:
        #     #     new_candidates.append(f'{cand1[2]} {cand2[2]}')
        #     # elif cand2[1] == cand1[0]:
        #     #     new_candidates.append(f'{cand2[2]} {cand1[2]}')
        #     if (cand1[1] + 1 == cand2[0]) and sentence.words[cand1[1]] in stop_words:
        #         stopword_separated_candidates.append(f'{cand1[2]} {sentence.words[cand1[1]]} {cand2[2]}')
        #         print(f'{cand1[2]} {sentence.words[cand1[1]]} {cand2[2]}')
        #     elif (cand2[1] + 1 == cand1[0]) and sentence.words[cand2[1]] in stop_words:
        #         stopword_separated_candidates.append(f'{cand2[2]} {sentence.words[cand2[1]]} {cand1[2]}')
        #         print(f'{cand2[2]} {sentence.words[cand2[1]]} {cand1[2]}')

        # Matching "findings" entities with consecutive candidates
        findings_candidates = [c for c in all_matches if 'T033' in umls_terminologies['others'][c]]
        findings_candidates.append(question_concept)
        elaborate_candidates = list(itertools.product(findings_candidates, consequent_candidates))
        
        all_matches_to_pair = []
        for concept in all_matches:
            if len(umls_terminologies['others'][concept].intersection(tuis_to_pair)) > 0:
                all_matches_to_pair.append(concept)
            else:
                non_symptom_candidates[concept.lower()] += 1

        all_matches_to_pair.append(question_concept)

        # All matches contains at least one symptom
        pair_candidates = list(itertools.combinations(all_matches_to_pair, 2)) + \
        list(itertools.product(all_matches_to_pair, (set(all_matches) - set(all_matches_to_pair))))

        candidates = list(set(all_matches) - remove_singleton) + pair_candidates + consequent_candidates + elaborate_candidates

        for cand in candidates:
            if isinstance(cand, tuple):
                cand1, cand2 = cand

                if (set(cand1.split(' ')).issubset(set(cand2.split(' '))) or set(cand2.split(' ')).issubset(set(cand1.split(' ')))):
                    print("Removed", cand)
                    candidates.remove(cand)

        row['candidates'] = candidates
        candidate_len.append(len(row['candidates']))
        rows.append(row)
        pbar.update(1)

    pbar.close()

    with open(pjoin('data', 'question_responses_results.json'), 'w') as f:
        json.dump(rows, f, indent=4)

    print(f"There were {unfound_q_concepts} question concepts not in the \"SnomedAttrType.FINDING\"")
    print(f"Number of concepts: mean={np.mean(candidate_len)}, max={np.max(candidate_len)}")
    non_symptom_candidates = sorted([x for x in non_symptom_candidates.items()], key=lambda x: x[1], reverse=True)
    # with open('non_symptom_candidates.txt', 'w') as f:
    #     for candidate in non_symptom_candidates:
    #         f.write(f'{candidate[0]}: {candidate[1]}\n')
    pdb.set_trace()

def construct_label_matrix(args, lfs, terminologies, umls_terminologies):
    pdb.set_trace()
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    dataset = load_json_dataset(args.jsonfile_path, tokenizer)
    X_sents = [dataset.sentences]
    labeler = SequenceLabelingServer(num_workers=4)

    # combined_terminologies = combine_ontologies(umls_terminologies, list(umls_terminologies.keys()))

    # L_sents = labeler.apply(lfs, X_sents)

    

    #     all_matches = [list(map(lambda x: x[1], lf(sentence, True, False)[1])) for lf in lfs]
    #     row['RFE'] = ' '.join(sentence.words)
        # for lf_idx in range(len(longest_matches)):
    #         lf_source_name = terminologies[lf_idx]
    #         row[lf_source_name] = {
    #             'all_matches': [],
    #             'longest_matches': [],
    #         }

    #         longest_matches_ = list(set(longest_matches[lf_idx]))
    #         all_matches_ = list(set(all_matches[lf_idx]))

    #         for term_idx, term in enumerate(longest_matches_):
    #             if term.lower() in umls_terminologies[lf_source_name].keys():
    #                 TUIs = umls_terminologies[terminologies[lf_idx]][term.lower()]
    #             elif term.rstrip('s').lower() in umls_terminologies[lf_source_name].keys():
    #                 TUIs = umls_terminologies[terminologies[lf_idx]][term.rstrip('s').lower()]

    #             row[lf_source_name]['longest_matches'].append({term: list(set(TUIs))})


    #         for term_idx, term in enumerate(all_matches_):
    #             if term.lower() in umls_terminologies[lf_source_name].keys():
    #                 TUIs = umls_terminologies[terminologies[lf_idx]][term.lower()]
    #             elif term.rstrip('s').lower() in umls_terminologies[lf_source_name].keys():
    #                 TUIs = umls_terminologies[terminologies[lf_idx]][term.rstrip('s').lower()]

    #             row[lf_source_name]['all_matches'].append({term: list(set(TUIs))})
    #     rows.append(row)

    # with open(pjoin('data', 'question_responses_results.json'), 'w') as f:
    #     json.dump(rows, f, indent=4)
    # pdb.set_trace()

    df = pd.DataFrame(rows)
    df.to_csv('data/question_responses_results.csv')
    pdb.set_trace()
    # df[1] = df[1].map(lambda x: len(x.split(',')))

    df[-1].map(lambda x: len(x.split(',')))

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
    pdb.set_trace()
    X_sents = X_sents[0]
    words = []
    prev_end_index = 0

    mv_extracted = 0

    
    for i, sent in enumerate(X_sents):
        y_pred_ = y_pred[prev_end_index: prev_end_index + len(sent.words)]
        L_train_ = L_train[prev_end_index: prev_end_index + len(sent.words)]

        y_pred_mv = np.zeros(len(sent.words))
        for j, votes in enumerate(L_train_):
            values, count = np.unique(votes, return_counts=True)
            max_count = count.max()

            if len(count[count == max_count]) > 1:
                mv = -1
            else:
                mv = values[count.argmax()]
            y_pred_mv[j] = mv

        mv_extracted += len(y_pred_mv[y_pred_mv == 1])

        row = []
        row.append(sent.words)

        # extracted_words = np.array(sent.words)[[L_train_ == 1][0].any(axis=1)]
        # for word_idx in range(len(extracted_words)):
        #     TUIs = combined_terminologies[word.lower()]
        # pdb.set_trace()
        # for lf_idx in range(L_words_hat.shape[-1]):
        #     extracted_words_lf = list(np.array(sent.words)[L_train_[:, lf_idx] == 1])

        #     for word_idx in range(len(extracted_words_lf)):
        #         word = extracted_words_lf[word_idx]
        #         TUIs = umls.terminologies[terminologies[lf_idx]][word.lower()]
        #         extracted_words_lf[word_idx] = f"{extracted_words_lf[word_idx]}: {TUIs}"
        #     row.append(extracted_words_lf)
            # extracted_words_lf = np.array(sent.words)[L_train_ == 1]

        # lf_predictions = [[] for _ in range(L_words_hat.shape[-1])]

        # for 
        # for sent_idx in range(len(L_sents[0])):
        #     L_sent = L_sents[0][sent_idx]
        #     for lf in L_sent

        extracted_words_mv = np.array(sent.words)[y_pred_mv == 1]
        row.append(extracted_words_mv)
        extracted_words_lfs = np.array(sent.words)[y_pred_ == 1]
        row.append(extracted_words_lfs)
        prev_end_index += len(sent.words)
        words.append(row) 

    print(f"MV coverage: {mv_extracted / len(y_pred)}")
    print(f"LM coverage: {len(y_pred[y_pred == 1]) / len(y_pred)}")
    df = pd.DataFrame(words)
    df[0] = df[0].map(lambda x: ' '.join(x))
    pdb.set_trace()

    lf_summary(L_words, Y=Y_words, lf_names=[lf.name for lf in lfs])
    df.to_csv('data/kb_responses_snorkle_results_v4.csv')
    

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
    parser.add_argument('-csvfile_path', default=pjoin(ROOT_DIR, 'data', 'bq-results-20211011-121939-k1658a4hpyvt.csv'))
    parser.add_argument('-jsonfile_path', default=pjoin(ROOT_DIR, 'data', 'kb_responses_full.json'))
    args = parser.parse_args()

    # install_from_zip(args)
    # test_installation(args)
    # preprocess_msgs(args)
    # preprocess_msgs_kb_responses(args)
    lfs, terminologies, umls_terminologies = create_lfs(args)
    get_entity_candidates(args, lfs, terminologies, umls_terminologies)
    # construct_label_matrix(args, lfs, terminologies, umls_terminologies)



