from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import tensorflow as tf
import util
import sys

import numpy as np

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

  model_name = sys.argv[1]
  
  
  trained_with_gold = True if 'gold' in model_name else False
  
  config = util.initialize_from_env()
  model = util.get_model(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  
  print_scores = True # if True it creates a test_scores.tsv file with evaluation scores
  all_masks = True # if True it goes through the same document multiple times to get masked predictions for all mentions
  with_gold_mentions = True # if True it also calculates predictions with gold mentions

  left_context_only_eval = True
  
  if not all_masks:
      eval_masked = 1 # iterations through masked evaluation
  
  if phase not in ['eval', 'test']:
    print('Specify a phase: test or eval')
    quit()
  
  if print_scores:
    output_file = open('data/' + model_name + '/' + phase + '_scores.tsv', 'w')
  else:
    output_file = None
      
  with tf.Session() as session:
    model.restore(session)
    
    # Make sure eval mode is True if you want official conll results
    # WARNING: in the masked evaluation conll score may differ from py ones due to skipped mentions (if included in masked mentions)
    
    
    if left_context_only_eval:
        print('Left-only: Evaluation without masks')
        # Non-masked evaluation; depending on with_gold_mentions with predicted and/or gold mentions
        model.evaluate(session, official_stdout=True, eval_mode=True, phase = phase, with_gold_mentions = with_gold_mentions, output_file = output_file, trained_with_gold = trained_with_gold, left_context_only = True, eval_on_masks_only = True )
        model.evaluate(session, official_stdout=True, eval_mode = True, masked = True, eval_on_masks_only = True, phase = phase, all_masks = False, with_gold_mentions = with_gold_mentions, output_file = output_file, trained_with_gold = trained_with_gold, left_context_only = True)
        
    
    print('Evaluation without masks')
    # Non-masked evaluation; depending on with_gold_mentions with predicted and/or gold mentions
    model.evaluate(session, official_stdout=True, eval_mode=True, phase = phase, with_gold_mentions = with_gold_mentions, output_file = output_file, trained_with_gold = trained_with_gold)
    
    if all_masks:
        # Masked evaluation; depending on with_gold_mentions with predicted and/or gold mentions
        model.evaluate(session, official_stdout=True, eval_mode = True, masked = True, eval_on_masks_only = True, phase = phase, all_masks = all_masks, with_gold_mentions = with_gold_mentions, output_file = output_file, trained_with_gold = trained_with_gold)
    else:
        # If not evaluation on all masked mentions
        i = 0
        
        eval_f1_mask_all, eval_antecedent_f1_mask_all,  eval_antecedent_f1_masks_only_all = [], [], []
        eval_f1_mask_all_gold, eval_antecedent_f1_mask_all_gold,  eval_antecedent_f1_masks_only_all_gold = [], [], []
        while i < eval_masked:
            print('Evaluation with masks - ', i)
            output_file.write('Run with masks - ' +  str(i) + '\n')
            output = model.evaluate(session, official_stdout=True, eval_mode = True, masked = True, eval_on_masks_only = True, phase = phase, with_gold_mentions = with_gold_mentions, output_file = output_file, trained_with_gold = trained_with_gold)
            if with_gold_mentions and not trained_with_gold:
                output_predicted, output_gold = output
            else:
                if trained_with_gold:
                    output_gold = output
                output_predicted = output
            
            if not trained_with_gold:
                eval_summary_mask, eval_f1_mask, eval_antecedent_f1_mask,  eval_antecedent_f1_masks_only = output_predicted
                eval_f1_mask_all.append(eval_f1_mask)
                eval_antecedent_f1_mask_all.append(eval_antecedent_f1_mask)
                eval_antecedent_f1_masks_only_all.append(eval_antecedent_f1_masks_only)
            
            if with_gold_mentions:
                eval_summary_mask, eval_f1_mask, eval_antecedent_f1_mask,  eval_antecedent_f1_masks_only = output_gold
                eval_f1_mask_all_gold.append(eval_f1_mask)
                eval_antecedent_f1_mask_all_gold.append(eval_antecedent_f1_mask)
                eval_antecedent_f1_masks_only_all_gold.append(eval_antecedent_f1_masks_only)
            
            i +=1
            
        if not trained_with_gold:
            eval_f1_mask = np.average(eval_f1_mask_all)
            eval_antecedent_f1_mask = np.average(eval_antecedent_f1_mask_all)
            eval_antecedent_f1_masks_only = np.average(eval_antecedent_f1_masks_only_all)
            
            print('With predicted mentions')
            print('Average F1 coreference with masks', eval_f1_mask)
            print('Average F1 antecedent with masks', eval_antecedent_f1_mask)
            print('Average F1 antecedent only on masks', eval_antecedent_f1_masks_only)
        
        eval_f1_mask = np.average(eval_f1_mask_all_gold)
        eval_antecedent_f1_mask = np.average(eval_antecedent_f1_mask_all_gold)
        eval_antecedent_f1_masks_only = np.average(eval_antecedent_f1_masks_only_all_gold)
        
        print('With gold mentions')
        print('Average F1 coreference with masks', eval_f1_mask)
        print('Average F1 antecedent with masks', eval_antecedent_f1_mask)
        print('Average F1 antecedent only on masks', eval_antecedent_f1_masks_only)
    

  output_file.close()
