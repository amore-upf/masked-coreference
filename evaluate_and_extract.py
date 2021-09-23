#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import tensorflow as tf
import util
import sys

import numpy as np

import json
from feature_extraction import features
import pandas as pd
from scipy.stats import entropy
import operator
import metrics
from independent import print_scores


def evaluate_and_save_predictions(model, session, global_step=None, official_stdout=False, keys=None, eval_mode=False, masked = False, eval_on_masks_only = False, selected_mentions = None, phase = 'eval', all_masks = False, save = False, left_context_only = False, with_gold_mentions = False, output_file = None, trained_with_gold = False):
    '''
    evaluation on all mentions and a subset of them +
    antecedent evaluation, saving info
    '''
    
    if left_context_only_eval:
        if masked:
            dataframe_file_predicted = open('output1/' + model_name +'_' + phase + '_with_predicted_mentions_left_context'+'_masks_dataframe.csv', 'w')
            dataframe_file_gold = open('output1/' + model_name +'_' + phase + '_with_gold_mentions_left_context'+'_masks_dataframe.csv', 'w')
        else:
            dataframe_file_predicted = open('output1/' + model_name +'_' + phase + '_with_predicted_mentions_left_context'+'_nomasks_dataframe.csv', 'w')
            dataframe_file_gold = open('output1/' + model_name +'_' + phase + '_with_gold_mentions_left_context'+'_nomasks_dataframe.csv', 'w')
    else:
        if masked:
            dataframe_file_predicted = open('output1/' + model_name +'_' + phase + '_with_predicted_mentions_whole_context'+'_masks_dataframe.csv', 'w')
            dataframe_file_gold = open('output1/' + model_name +'_' + phase + '_with_gold_mentions_whole_context'+'_masks_dataframe.csv', 'w')
        else:
            dataframe_file_predicted = open('output1/' + model_name +'_' + phase + '_with_predicted_mentions_whole_context'+'_nomasks_dataframe.csv', 'w')
            dataframe_file_gold = open('output1/' + model_name +'_' + phase + '_with_gold_mentions_whole_context'+'_nomasks_dataframe.csv', 'w')
    
    if masked:
        if left_context_only:
            model.eval_data_masks = None
        else:
            model.eval_data_left_only_masks = None
            
    if phase == 'test':
        model.eval_data = None
        model.eval_data_left_only_masks = None
        
    model.load_eval_data(masked = masked, phase = phase, all_masks = all_masks, left_context_only = left_context_only)
    
    if not left_context_only:
        if masked:
            eval_data = model.eval_data_masks
        else:
            eval_data = model.eval_data
    else:
        if masked:
            eval_data = model.eval_data_left_only_masks
        else:
            eval_data = model.eval_data_left_only
                
    if phase == 'test' and save:
        if left_context_only:
            saved_input_filename = phase + '_leftcontext.json'
        else:
            saved_input_filename = phase + '_wholecontext.json'
        with open(saved_input_filename, 'w') as saved_inputs:
            for (_, example) in eval_data:
                saved_inputs.write(example + '\n')
    
    def extract_and_compute_evaluation_scores(eval_data, dataframe_file, with_gold_mentions = False):
    
        doc = []
        sentence = []
        genre = []
        mention = []
        predicted_antecedent = []
        gold_antecedent = []
        gold_antecedent_masked_not = []
        predicted_antecedent_masked_not = []
        antecedent_prediction_quality = []
        masked_or_not = []
        original_indices = []
        gold_antecedent_original_indices = []
        number_of_masked_mentions = []
        masked_mentions = []
        probability_of_gold_antecedent = []
        probability_of_gold_entity = []
        probability_of_predicted_antecedent = []
        probability_of_predicted_entity = []
        entropy_of_probability_distribution = []
        entropy_distribution_over_entities = []
        entropy_other_entities = []
        surprisal_of_gold_antecedent = []
        surprisal_of_gold_entity = []
        strings_of_mention = []
        mention_length = []
        mention_type = []
        antecedent_type = []
        mention_grammatical_function = []
        antecedent_grammatical_function = []
        mention_antecedent_sentence_distance = []
        distance_tokens = []
        distractive_mentions = []
        distractive_entities = []
        previous_tokens = []
        previous_mentions = []
        previous_entities = []
        prior_reference_times_entity = []
        entity_topicality = []
        entity_topicality_by_tokens = []
        masked_rows = []
        masked_row_doc_keys = []
        masked_row_predicted_antecedent = []
        
        mappings_original = {}
        subtoken_maps_original = {}
        mappings_tokens_original = {}
        subtoken_maps_original = {}
        
        losses = []
        doc_keys = []
        num_evaluated= 0
        
        target_rows = []
        target_row_doc_keys = []
        target_row_predicted_antecedent = []
        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()
        antecedent_evaluator = metrics.CorefEvaluator_antecedent() 
        #    coref_evaluator_selected = metrics.CorefEvaluator()
            antecedent_evaluator_selected =  metrics.CorefEvaluator_antecedent()
        else:
        #    coref_evaluator_selected = None
            antecedent_evaluator_selected = None
        losses = []
        doc_keys = []
        num_evaluated= 0
    
        
        mappings_original = {}
        subtoken_maps_original = {}
        mappings_tokens_original = {}
        for example_num, (tensorized_example, example) in enumerate(eval_data):
            _, _, _, _, _, _, gold_starts, gold_ends, _, _ = tensorized_example
            tensorized_example = list(tensorized_example)
            tensorized_example.append(with_gold_mentions)
            tensorized_example = tuple(tensorized_example)
            if masked:
                mappings_original[example['doc_key']] = example['from_new_to_original_indices']
                subtoken_maps_original[example['doc_key']] = example['original_subtoken_map']
                mappings_tokens_original[example['doc_key']] = example['from_new_to_original_indices_tokens']
            feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
            # if tensorized_example[0].shape[0] <= 9:
            if keys is not None and example['doc_key'] not in keys:
            # print('Skipping...', example['doc_key'], tensorized_example[0].shape)
                continue
            doc_keys.append(example['doc_key'])
            loss, (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores) = session.run([model.loss, model.predictions], feed_dict=feed_dict)
            losses.append(loss)
            predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
                
            if eval_on_masks_only:
                if masked:
                    selected_mentions = example['masked_mentions']
                else:
                    selected_mentions = example['target_mention']
            else:
                selected_mentions = None
            #else:
            #    selected_mentions = model.masks_used[example_num]
                
            coref_predictions[example["doc_key"]] = model.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator) #, selected_mentions = selected_mentions, evaluator_selected = coref_evaluator_selected)

            gold_clusters = [tuple(tuple(m) for m in gc) for gc in example['clusters']]
            antecedent_evaluate_update = model.evaluate_antecedent(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], antecedent_evaluator, selected_mentions = selected_mentions, evaluator_selected  = antecedent_evaluator_selected)

            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(eval_data)))

            # EXTRACT FEATURES
            gold_clusters = [tuple(tuple(m) for m in gc) for gc in example["clusters"]]
            mention_to_gold = {}
            first_mentions = []
            for gc in gold_clusters:
                if exclude_first_mention:
                    first_mention = sorted(gc, key= lambda x: x[0])[0]
                    first_mentions.append(first_mention)
                    for m in gc:
                        if m != first_mention:
                        # specify subset of mentions here
                            mention_to_gold[m] = gc
                else:
                    for m in gc:
                        mention_to_gold[m] = gc
                    
            mention_to_predicted_antecedent = model.get_mention_top_predicted_antecedents(top_span_starts, top_span_ends, predicted_antecedents, mention_to_gold.keys())

            
            if left_context_only:
                if not masked:
                    target_mentions = [tuple(m) for m in example['target_mention']] #for left context only
                else:
                    target_mentions = [tuple(m) for m in example['masked_mentions']]
            else:
                if not masked:
                    target_mentions = list(mention_to_predicted_antecedent.keys())
                else:
                    target_mentions = [tuple(m) for m in example['masked_mentions']]
            
            sub_mention = target_mentions
            
            mention_to_predicted_antecedent_target = {key: mention_to_predicted_antecedent[key] for key in mention_to_predicted_antecedent if key in sub_mention} # only those that are to be evaluated
            
            sub_mention = [m for m in sub_mention if m in mention_to_predicted_antecedent.keys()] # only those that are to be evaluated and actually detected
            
            if len(sub_mention) > 0:
                sub_doc = [example['doc_key']] * len(sub_mention)
                sub_predicted_antecedent = list(mention_to_predicted_antecedent_target.values())
                mention.extend(sub_mention)
                doc.extend(sub_doc)
                predicted_antecedent.extend(sub_predicted_antecedent)
                
                extractor = features(conll_formatted_ontonotes_path, example, sub_mention)
                example["antecedent_prediction_quality"] = extractor.correct_wrong_prediction(mention_to_predicted_antecedent_target, mention_to_gold)
                example["mention_to_gold_antecedent"] = extractor.mention_last_antecedent()
                if masked:
                    example["mention_to_gold_antecedent"] = extractor.mention_last_antecedent()
                    example["gold_antecedent_masked_or_not"] = extractor.gold_antecedent_masked_or_not(example["mention_to_gold_antecedent"], example['masked_mentions'])
                    example["masked_or_not"] = extractor.mention_masked_or_not ()
                    example["original_indices"] = extractor.new_to_original_indices ()
                mention_strings = extractor.mention_to_string_mention()
                mention_length_token = extractor.mention_token_length()
                mention_to_sentence = extractor.mention_to_sentence ()
                example["mention_type"] = extractor.mention_to_mentiontype()
                example["antecedent_type"] = extractor.antecedent_type()
                grammatical_function_mention = extractor.mention_grammatical_function()
                example["antecedent_grammatical_function"] = extractor.antecedent_grammatical_function()
                example["mention_antecedent_sentence_distance"]= extractor.mention_antecedent_sentence_distance()
                example["distance_tokens"] = extractor.distance_to_last_antecedent()
                example["distractive_mentions"] = extractor.number_of_distractors()
                example["distractive_entities"] = extractor.distractive_entities()
                example["previous_mentions"] = extractor.number_of_previous_mentions()
                example["previous_entities"] = extractor.number_of_previous_entities()
                number_previous_tokens = extractor.mention_previous_tokens()
                mention_genre = extractor.mention_to_genre()
                example["prior_reference_times_entity"] = extractor.entity_prior_reference_frequency()
                example["entity_topicality"] = extractor.entity_topicality()

                np.set_printoptions(threshold=np.inf)
                np.set_printoptions(suppress=True)
                pro_dist = extractor.probability_distribution(sub_mention, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores)
                pro_gold_antecedent = extractor.probability_gold_antecedent(example["mention_to_gold_antecedent"], pro_dist)
                pro_gold_entity = extractor.probability_gold_entity(mention_to_gold, pro_dist)
                pro_predicted_antecedent = extractor.probability_predicted_antecedent(mention_to_predicted_antecedent_target, pro_dist)
                pro_predicted_entity = extractor.probability_predicted_entity(mention_to_predicted_antecedent_target, pro_dist)
                entropy_of_distribution = extractor.entropy_distribution(pro_dist)
                entropy_of_distribution_over_entities = extractor.entropy_distribution_over_entities(pro_dist)
                entropy_of_other_entities = extractor.entropy_distribution_over_entities_excluding_gold_antecedent (sub_mention, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores)
                dict_surprisal_gold_antecedent = extractor.surprisal_gold_antecedent(pro_gold_antecedent)
                dict_surprisal_gold_entity = extractor.surprisal_gold_entity(pro_gold_entity[0])
                
                for i,row in enumerate(sub_mention):
                    target_rows.append(row)
                    target_row_doc_keys.append(sub_doc[i])
                    target_row_predicted_antecedent.append(sub_predicted_antecedent[i])
                    antecedent_prediction_quality.append(example["antecedent_prediction_quality"][row])
                    gold_antecedent.append(example["mention_to_gold_antecedent"][row])
                    sentence.append(mention_to_sentence[row][0])
                    mention_type.append(example["mention_type"][str(row)])
                    antecedent_type.append(example["antecedent_type"][str(row)])
                    mention_grammatical_function.append(grammatical_function_mention[row])
                    antecedent_grammatical_function.append(example["antecedent_grammatical_function"][str(row)])
                    mention_antecedent_sentence_distance.append(example["mention_antecedent_sentence_distance"][str(row)])
                    distance_tokens.append(example["distance_tokens"][str(row)])
                    distractive_mentions.append(example["distractive_mentions"][str(row)])
                    distractive_entities.append(example["distractive_entities"][str(row)])
                    previous_mentions.append(example["previous_mentions"][str(row)])
                    previous_entities.append(example["previous_entities"][str(row)])
                    previous_tokens.append(number_previous_tokens[row])
                    genre.append(mention_genre[row])
                    prior_reference_times_entity.append(example["prior_reference_times_entity"][str(row)])
                    entity_topicality.append(example["entity_topicality"][str(row)])
                    entity_topicality_by_tokens.append(extractor.entity_topicality_by_previous_tokens()[row])
                    if masked:
                        if sub_predicted_antecedent[i] in example["masked_mentions"]:
                            predicted_antecedent_masked_not.append("masked")
                        else:
                            predicted_antecedent_masked_not.append("unmasked")
                        gold_antecedent_masked_not.append(example["gold_antecedent_masked_or_not"][row])
                        masked_or_not.append(example["masked_or_not"][str(row)])
                        original_indices.append(example["original_indices"][str(row)])
                        gold_antecedent_original_indices.append(example["original_indices"][str(example["mention_to_gold_antecedent"][row])])
                        number_of_masked_mentions.append(len(example["masked_mentions"]))
                        masked_mentions.append(example["masked_mentions"])
                    
                    probability_of_gold_antecedent.append(pro_gold_antecedent[row])
                    probability_of_gold_entity.append(pro_gold_entity[0][row])
                    probability_of_predicted_antecedent.append(pro_predicted_antecedent[row])
                    probability_of_predicted_entity.append(pro_predicted_entity[0][row])
                    entropy_of_probability_distribution.append(entropy_of_distribution[0][row])
                    entropy_distribution_over_entities.append(entropy_of_distribution_over_entities[row])
                    entropy_other_entities.append(entropy_of_other_entities[row])
                    surprisal_of_gold_antecedent.append(dict_surprisal_gold_antecedent[row])
                    surprisal_of_gold_entity.append(dict_surprisal_gold_entity[row])
                    strings_of_mention.append(mention_strings[str(row)])
                    mention_length.append(mention_length_token[row])

        summary_dict = {}
        if output_file != None:
            print_scores(output_file, coref_evaluator, antecedent_evaluator, antecedent_evaluator_selected, with_gold_mentions, masked)
            
        p,r,f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        antecedent_p, antecedent_r, antecedent_f = antecedent_evaluator.get_prf()
        summary_dict["Average antecedent_F1 (py)"] = antecedent_f
        print("Average antecedent_F1 (py): {:.2f}% on {} docs".format(antecedent_f * 100, len(doc_keys)))
        summary_dict["Average antecedent_precision (py)"] = antecedent_p
        print("Average antecedent_precision (py): {:.2f}%".format(antecedent_p * 100))
        summary_dict["Average antecedent_recall (py)"] = antecedent_r
        print("Average antecedent_recall (py): {:.2f}%".format(antecedent_r * 100))
        
        if eval_on_masks_only:
            '''
            p_masks, r_masks, f_masks = coref_evaluator_selected.get_prf()
            summary_dict["Average masks F1 (py)"] = f_masks
            print("Average masks F1 (py): {:.2f}% on {} docs".format(f_masks * 100, len(doc_keys)))
            summary_dict["Average masks precision (py)"] = p_masks
            print("Average masks precision (py): {:.2f}%".format(p_masks * 100))
            summary_dict["Average masks recall (py)"] = r_masks
            print("Average masks recall (py): {:.2f}%".format(r_masks * 100))
            '''
            antecedent_p_masks, antecedent_r_masks, antecedent_f_masks =  antecedent_evaluator_selected.get_prf()
            summary_dict["Average masks antecedent_F1 (py)"] = antecedent_f_masks
            print("Average masks antecedent_F1 (py): {:.2f}% on {} docs".format(antecedent_f_masks * 100, len(doc_keys)))
            summary_dict["Average masks antecedent_precision (py)"] = antecedent_p_masks
            print("Average masks antecedent_precision (py): {:.2f}%".format(antecedent_p_masks * 100))
            summary_dict["Average masks antecedent_recall (py)"] = antecedent_r_masks
            print("Average masks antecedent_recall (py): {:.2f}%".format(antecedent_r_masks * 100))

            
        if not masked:
            d = {'doc': target_row_doc_keys, 'sentence': sentence, 'genre': genre, 'mention': target_rows, 'gold_antecedent': gold_antecedent,  'predicted_antecedent': target_row_predicted_antecedent, 'antecedent_prediction_quality': antecedent_prediction_quality,  'probability_of_gold_antecedent': probability_of_gold_antecedent, 'probability_of_gold_entity': probability_of_gold_entity,  'probability_of_predicted_antecedent': probability_of_predicted_antecedent, 'probability_of_predicted_entity': probability_of_predicted_entity,'entropy_of_probability_distribution':entropy_of_probability_distribution, 'entropy_distribution_over_entities': entropy_distribution_over_entities, 'entropy_other_entities': entropy_other_entities,  'surprisal_of_gold_antecedent': surprisal_of_gold_antecedent,  'surprisal_of_gold_entity': surprisal_of_gold_entity, 'strings_of_mention': strings_of_mention, 'mention_length': mention_length,  'mention_type': mention_type, 'antecedent_type': antecedent_type, 'mention_grammatical_function': mention_grammatical_function, 'antecedent_grammatical_function': antecedent_grammatical_function, 'mention_antecedent_sentence_distance': mention_antecedent_sentence_distance, 'distance_tokens': distance_tokens, 'distractive_mentions': distractive_mentions, 'distractive_entities': distractive_entities, 'previous_mentions': previous_mentions, 'previous_entities': previous_entities, 'previous_tokens': previous_tokens, 'prior_reference_times_entity': prior_reference_times_entity, 'entity_topicality': entity_topicality, 'entity_topicality_by_tokens':entity_topicality_by_tokens}
            
            
            df = pd.DataFrame(data=d, columns = ['doc','sentence', 'genre', 'mention','gold_antecedent','predicted_antecedent',  'antecedent_prediction_quality',
                             'probability_of_gold_antecedent', 'probability_of_gold_entity', 'probability_of_predicted_antecedent',
                             'probability_of_predicted_entity','entropy_of_probability_distribution','entropy_distribution_over_entities','entropy_other_entities', 'surprisal_of_gold_antecedent','surprisal_of_gold_entity','strings_of_mention','mention_length', 'mention_type', 'antecedent_type', 'mention_grammatical_function', 'antecedent_grammatical_function',
                                 'mention_antecedent_sentence_distance', 'distance_tokens', 'distractive_mentions', 'distractive_entities',
                                 'previous_mentions', 'previous_entities', 'previous_tokens', 'prior_reference_times_entity',
                                 'entity_topicality', 'entity_topicality_by_tokens'])
        else:
            d = {'doc': target_row_doc_keys, 'genre': genre, 'sentence': sentence, 'mention': target_rows, 'gold_antecedent': gold_antecedent,'original_indices_gold_antecedent':gold_antecedent_original_indices, 'gold_antecedent_masked_or_not': gold_antecedent_masked_not,
         'predicted_antecedent': target_row_predicted_antecedent, 'predicted_antecedent_masked_or_not': predicted_antecedent_masked_not, 'antecedent_prediction_quality': antecedent_prediction_quality,
         'masked_or_not': masked_or_not, 'original_indices': original_indices,'number_of_masked_mentions': number_of_masked_mentions, 'masked_mentions': masked_mentions,
         'probability_of_gold_antecedent': probability_of_gold_antecedent, 'probability_of_gold_entity': probability_of_gold_entity,
         'probability_of_predicted_antecedent': probability_of_predicted_antecedent, 'probability_of_predicted_entity': probability_of_predicted_entity,
         'entropy_of_probability_distribution':entropy_of_probability_distribution, 'entropy_distribution_over_entities':entropy_distribution_over_entities,'entropy_other_entities': entropy_other_entities, 'surprisal_of_gold_antecedent': surprisal_of_gold_antecedent,
         'surprisal_of_gold_entity': surprisal_of_gold_entity, 'strings_of_mention': strings_of_mention,'mention_type': mention_type, 'mention_length': mention_length, 'antecedent_type': antecedent_type,
         'mention_grammatical_function': mention_grammatical_function, 'antecedent_grammatical_function': antecedent_grammatical_function, 'mention_antecedent_sentence_distance': mention_antecedent_sentence_distance,
             'distance_tokens': distance_tokens, 'distractive_mentions': distractive_mentions,
             'distractive_entities': distractive_entities, 'previous_mentions': previous_mentions,
             'previous_entities': previous_entities, 'previous_tokens': previous_tokens,
             'prior_reference_times_entity': prior_reference_times_entity,
             'entity_topicality': entity_topicality, 'entity_topicality_by_tokens':entity_topicality_by_tokens}
             
            
            df = pd.DataFrame(data=d, columns = ['doc','sentence', 'genre','mention','gold_antecedent','original_indices_gold_antecedent', 'gold_antecedent_masked_or_not','predicted_antecedent',
                         'predicted_antecedent_masked_or_not','antecedent_prediction_quality',
                         'masked_or_not', 'original_indices', 'number_of_masked_mentions', 'masked_mentions',
                         'probability_of_gold_antecedent', 'probability_of_gold_entity','probability_of_predicted_antecedent',
                           'probability_of_predicted_entity','entropy_of_probability_distribution','entropy_distribution_over_entities',
                         'entropy_other_entities', 'surprisal_of_gold_antecedent', 'surprisal_of_gold_entity','strings_of_mention','mention_length',
                         'mention_type', 'antecedent_type', 'mention_grammatical_function', 'antecedent_grammatical_function',
                             'mention_antecedent_sentence_distance', 'distance_tokens',
                             'distractive_mentions', 'distractive_entities',
                             'previous_mentions', 'previous_entities', 'previous_tokens', 'prior_reference_times_entity',
                             'entity_topicality', 'entity_topicality_by_tokens'])
        
        df.to_csv(dataframe_file, index = True, header=True)
        dataframe_file.close
            
    extract_and_compute_evaluation_scores(eval_data, dataframe_file_predicted, with_gold_mentions = False)
    extract_and_compute_evaluation_scores(eval_data, dataframe_file_gold, with_gold_mentions = True)


def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
   
    
    try:
        phase = sys.argv[2]
    except:
        phase = 'test'

    if phase not in ['eval', 'test']:
        print('Specify a phase: test or eval')
        quit()
  
    try:
        left_context_only_eval = sys.argv[3]
    except:
        left_context_only_eval = False
    
    model_name = sys.argv[1]

    config = util.initialize_from_env()
    model = util.get_model(config)
    saver = tf.train.Saver()
    log_dir = config["log_dir"]

    conll_formatted_ontonotes_path = config["conll_formatted_ontonotes_path"]


    print_eval_scores = True # if True it creates a test_scores.tsv file with evaluation scores
    all_masks = True # if True it goes through the same document multiple times to get masked predictions for all mentions
    with_gold_mentions = True # if True it also calculates predictions with gold mentions

    #left_context_only_eval = False
    exclude_first_mention = False
    # Make sure eval mode is True if you want official conll results
    # WARNING: in the masked evaluation conll score may differ from py ones due to skipped mentions (if included in masked mentions)
           

    if print_eval_scores:
        context = 'left' if left_context_only_eval else 'whole'
        output_file = open('data/' + model_name + '/' + phase + '_' + context + 'context_scores.tsv', 'w')
    else:
        output_file = None
      
    with tf.Session() as session:
        model.restore(session)
        if left_context_only_eval:
            print('Left-only')

        if left_context_only_eval:
            print('Evaluation with left context, without masks')
            evaluate_and_save_predictions(model, session, official_stdout=True, eval_mode=True, phase = phase, with_gold_mentions = with_gold_mentions, output_file = output_file, left_context_only = True, eval_on_masks_only = True, all_masks = False) # left context
            print('Evaluation with left context, with masks')
            evaluate_and_save_predictions(model, session, official_stdout=True, eval_mode=True, phase = phase, with_gold_mentions = with_gold_mentions, output_file = output_file, left_context_only = True, eval_on_masks_only = True, masked = True, all_masks = False) # left context
        else:
            print('Evaluation with whole context, without masks')
            evaluate_and_save_predictions(model, session, official_stdout=True, eval_mode=True, phase = phase, with_gold_mentions = with_gold_mentions, output_file = output_file, left_context_only = False, eval_on_masks_only = False, all_masks = False) # whole context
            print('Evaluation with whole context, with masks')
            evaluate_and_save_predictions(model, session, official_stdout=True, eval_mode=True, phase = phase, with_gold_mentions = with_gold_mentions, output_file = output_file, left_context_only = False, eval_on_masks_only = True, masked = True, all_masks = True) # left context
                            

        if print_eval_scores:
            output_file.close()
