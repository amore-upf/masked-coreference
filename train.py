from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import torch
import tensorflow as tf
import util
import logging
import numpy as np

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
  config = util.initialize_from_env()

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]
  eval_masked = 5
  #eval_frequency = 100 # for debugging

  model = util.get_model(config)
  saver = tf.train.Saver()

  log_dir = config["log_dir"]
  max_steps = config['num_epochs'] * config['num_docs']
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  mode = 'w'
  max_antecedent_f1 = 0 # added
  
  masked = config['mask_percentage'] > 0

  with tf.Session() as session:
    # Initialize
    session.run(tf.global_variables_initializer())
    # load_data altogether
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
      mode = 'a'
    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    initial_time = time.time()
    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step  > 0 and tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        
        '''
        if max_percentage > 0: evaluation on masks only too
        Stopping criterion: model with max sum of both accuracies
        '''
        
        i = 0
        eval_f1_mask_all, eval_antecedent_f1_mask_all,  eval_antecedent_f1_masks_only_all = [], [], []
        while i < eval_masked:
            print('Evaluation with masks - ', i)
            eval_summary_mask, eval_f1_mask, eval_antecedent_f1_mask,  eval_antecedent_f1_masks_only = model.evaluate(session, tf_global_step, masked = True, eval_on_masks_only = True,
                                with_gold_mentions = config['with_gold_mentions'])
            eval_f1_mask_all.append(eval_f1_mask)
            eval_antecedent_f1_mask_all.append(eval_antecedent_f1_mask)
            eval_antecedent_f1_masks_only_all.append(eval_antecedent_f1_masks_only)
            writer.add_summary(eval_summary_mask, tf_global_step)
            i +=1
            
        eval_f1_mask = np.average(eval_f1_mask_all)
        eval_antecedent_f1_mask = np.average(eval_antecedent_f1_mask_all)
        eval_antecedent_f1_masks_only = np.average(eval_antecedent_f1_masks_only_all)
            
        # keep model with max f1 mask
        global_eval_f1 = eval_f1_mask
          
        print('Evaluation without masks')
        eval_summary, eval_f1, eval_antecedent_f1 = model.evaluate(session, tf_global_step, with_gold_mentions = config['with_gold_mentions'])

        if not masked:
                global_eval_f1 = eval_f1
        writer.add_summary(eval_summary, tf_global_step)
        
        if global_eval_f1 > max_f1:
          max_f1 = global_eval_f1
          max_scores = {'coref_f1': eval_f1, 'antecedent_f1': eval_antecedent_f1, 'coref_withmasks_f1': eval_f1_mask, 'antecedent_withmasks_f1': eval_antecedent_f1_mask, 'antecedent_onlymasks_f1': eval_antecedent_f1_masks_only}
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
         
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))
        logger.info("[{}] evaL_antecedent_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_antecedent_f1, max_f1))
        logger.info("[{}] evaL_mask_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1_mask, max_f1))
        logger.info("[{}] evaL_mask_antecedent_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_antecedent_f1_mask, max_f1))
        logger.info("[{}] evaL_mask_antecedent_f1_selected={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_antecedent_f1_masks_only, max_f1))
        if tf_global_step > max_steps:
                logger.info(str(max_scores))
                break
          
