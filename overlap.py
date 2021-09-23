from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf

import util
import coref_ops
import conll
import metrics
import optimization
from bert import tokenization
from bert import modeling
from pytorch_to_tf import load_from_pytorch_checkpoint

class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.subtoken_maps = {}
    self.max_segment_len = config['max_segment_len']
    self.max_span_width = config["max_span_width"]
    self.mask_perc = config['mask_percentage'] #MODIFIED
    self.n_placeholders =  1 #MODIFIED
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    self.eval_data = None # Load eval data lazily.
    self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
    self.sep = 102
    self.cls = 101
    self.tokenizer = tokenization.FullTokenizer(
                vocab_file=config['vocab_file'], do_lower_case=False)

    input_props = []
    input_props.append((tf.int32, [None, None])) # input_ids.
    input_props.append((tf.int32, [None, None])) # input_mask
    input_props.append((tf.int32, [None, None])) # input_ids.
    input_props.append((tf.int32, [None, None])) # input_mask
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.int32, [None, None])) # Speaker IDs.
    input_props.append((tf.int32, [])) # Genre.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None])) # Cluster ids.
    input_props.append((tf.int32, [None])) # Sentence Map

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()
    
    
    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    # bert stuff
    tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config['tf_checkpoint'])
    init_from_checkpoint = tf.train.init_from_checkpoint if config['init_checkpoint'].endswith('ckpt') else load_from_pytorch_checkpoint
    init_from_checkpoint(config['init_checkpoint'], assignment_map)
    print("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      # init_string)
      print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

    num_train_steps = int(
                    self.config['num_docs'] * self.config['num_epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    self.global_step = tf.train.get_or_create_global_step()
    self.train_op = optimization.create_custom_optimizer(tvars,
                      self.loss, self.config['bert_learning_rate'], self.config['task_learning_rate'],
                      num_train_steps, num_warmup_steps, False, self.global_step, freeze=-1)

  def start_enqueue_thread(self, session):
    print('Loading data')
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        if self.config['single_example']:
          for example in train_examples:
            example = add_masks(example, mask_profile = 'percentage', all_profiles = False, n_masks_profile = 1, skip_first_mention = False,
                                perc_mask=self.mask_perc, n_placeholders = self.n_placeholders)
            tensorized_example = self.tensorize_example(example[0], is_training=True)
            feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
            session.run(self.enqueue_op, feed_dict=feed_dict)
        else:
          examples = []
          for example in train_examples:
            example = add_masks(example, mask_profile = 'percentage', all_profiles = False, n_masks_profile = 1, skip_first_mention = False,
                                perc_mask=self.mask_perc, n_placeholders = self.n_placeholders)
            tensorized = self.tensorize_example(example[0], is_training=True)
            if type(tensorized) is not list:
              tensorized = [tensorized]
            examples += tensorized
          random.shuffle(examples)
          print('num examples', len(examples))
          for example in examples:
            feed_dict = dict(zip(self.queue_input_tensors, example))
            session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() ]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)


  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_span_labels(self, tuples, label_dict):
    if len(tuples) > 0:
      starts, ends, labels = zip(*tuples)
    else:
      starts, ends, labels = [], [], []
    return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

  def get_speaker_dict(self, speakers):
    speaker_dict = {'UNK': 0, '[SPL]': 1}
    for s in speakers:
      if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
        speaker_dict[s] = len(speaker_dict)
    return speaker_dict


  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]
    #sentences = [sentence[1:-1] for sentence in sentences]
    num_words = sum(len(s) for s in sentences)
    speakers = example["speakers"]
    # assert num_words == len(speakers), (num_words, len(speakers))
    speaker_dict = self.get_speaker_dict(util.flatten(speakers))
    sentence_map = example['sentence_map']


    max_sentence_length = self.max_segment_len #270 #max(len(s) for s in sentences)
    max_len = max(len(s) + 2 for s in sentences) # CHANGED; two symbols added later
    if max_len > max_sentence_length:
        max_sentence_length = max_len
    text_len = np.array([len(s) for s in sentences])

    input_ids, input_mask, speaker_ids, prev_overlap = [], [], [], []
    overlap_ids, overlap_mask = [], []
    half = self.max_segment_len // 2
    prev_tokens_per_seg = []
    for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
      prev_tokens_per_seg += [len(prev_overlap)]
      overlap_words = ['[CLS]'] + prev_overlap + sentence[:half] + ['[SEP]']
      prev_overlap = sentence[half:]
      sentence = ['[CLS]'] + sentence + ['[SEP]']
      sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
      sent_input_mask = [1] * len(sent_input_ids)
      sent_speaker_ids = [speaker_dict.get(s, 0) for s in ['##'] + speaker + ['##']]
      while len(sent_input_ids) < max_sentence_length:
          sent_input_ids.append(0)
          sent_input_mask.append(0)
          sent_speaker_ids.append(0)
      overlap_input_ids = self.tokenizer.convert_tokens_to_ids(overlap_words)
      overlap_input_mask = [1] * len(overlap_input_ids)
      while len(overlap_input_ids) < max_sentence_length:
        overlap_input_ids.append(0)
        overlap_input_mask.append(0)
      input_ids.append(sent_input_ids)
      speaker_ids.append(sent_speaker_ids)
      input_mask.append(sent_input_mask)
      overlap_ids.append(overlap_input_ids)
      overlap_mask.append(overlap_input_mask)
    overlap_words = ['[CLS]'] + prev_overlap + ['[SEP]']
    overlap_input_ids = self.tokenizer.convert_tokens_to_ids(overlap_words)
    overlap_input_mask = [1] * len(overlap_input_ids)
    prev_tokens_per_seg += [len(prev_overlap)]
    while len(overlap_input_ids) < max_sentence_length:
      overlap_input_ids.append(0)
      overlap_input_mask.append(0)
    overlap_ids.append(overlap_input_ids)
    overlap_mask.append(overlap_input_mask)
    input_ids = np.array(input_ids)
    input_mask = np.array(input_mask)
    speaker_ids = np.array(speaker_ids)
    overlap_ids = np.array(overlap_ids)
    overlap_mask = np.array(overlap_mask)
    assert num_words == (np.sum(input_mask) - 2*np.shape(input_mask)[0]), (num_words, np.sum(input_mask))
    assert num_words == (np.sum(overlap_mask) - 2*np.shape(overlap_mask)[0]), (num_words, np.sum(overlap_mask), np.shape(overlap_mask))


    doc_key = example["doc_key"]
    self.subtoken_maps[doc_key] = example["subtoken_map"]
    genre = self.genres[doc_key[:2]]

    gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
    example_tensors = (input_ids, input_mask, overlap_ids, overlap_mask,  text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map)

    if is_training and len(sentences) > self.config["max_training_sentences"]:
      return self.truncate_example(* (example_tensors + (prev_tokens_per_seg, )))
    else:
      return example_tensors

  def truncate_example(self, input_ids, input_mask, overlap_ids, overlap_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, prev_tokens_per_seg, sentence_offset=None):
    max_training_sentences = self.config["max_training_sentences"]
    num_sentences = input_ids.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    overlap_ids = overlap_ids[sentence_offset:sentence_offset + max_training_sentences + 1, :]
    overlap_mask = overlap_mask[sentence_offset:sentence_offset + max_training_sentences + 1, :]
    overlap_ids[-1, 1 + prev_tokens_per_seg[sentence_offset + max_training_sentences]] = self.sep
    overlap_ids[-1, 2 + prev_tokens_per_seg[sentence_offset + max_training_sentences]:] = 0
    overlap_mask[-1, 2 + prev_tokens_per_seg[sentence_offset + max_training_sentences]:] = 0
    overlap_mask[0, 1: 1 + prev_tokens_per_seg[sentence_offset ]] = 0
    assert num_words == overlap_mask.sum() - 2 * np.shape(overlap_ids)[0], (num_words, overlap_mask.sum(), text_len)
    speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    sentence_map = sentence_map[word_offset: word_offset + num_words]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    return input_ids, input_mask, overlap_ids, overlap_mask, text_len, speaker_ids, genre, is_training,  gold_starts, gold_ends, cluster_ids, sentence_map

  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span)) # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0) # [num_candidates]
    return candidate_labels

  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]
    if self.config['use_prior']:
      antecedent_distance_buckets = self.bucket_distance(antecedent_offsets) # [k, c]
      distance_scores = util.projection(tf.nn.dropout(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), self.dropout), 1, initializer=tf.truncated_normal_initializer(stddev=0.02)) #[10, 1]
      antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1), antecedent_distance_buckets) # [k, c]
      fast_antecedent_scores += antecedent_distance_scores

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

  def combine_passes(self, original_doc, input_ids, input_mask, overlap_doc, overlap_ids, overlap_mask):
    overlap_mask, input_mask = tf.equal(overlap_mask, 1), tf.equal(input_mask, 1)
    org_content_mask = tf.logical_and(input_mask, tf.logical_and(tf.not_equal(input_ids, self.cls), tf.not_equal(input_ids, self.sep)))
    overlap_content_mask = tf.logical_and(overlap_mask, tf.logical_and(tf.not_equal(overlap_ids, self.cls), tf.not_equal(overlap_ids, self.sep)))
    flat_org_doc = self.flatten_emb_by_sentence(original_doc, org_content_mask)
    flat_overlap_doc = self.flatten_emb_by_sentence(overlap_doc, overlap_content_mask)
    with tf.variable_scope("combo"):
      f = tf.sigmoid(util.projection(tf.concat([flat_org_doc, flat_overlap_doc], -1), util.shape(flat_org_doc, -1))) # [n, emb]
      combo = f * flat_org_doc + (1 - f) * flat_overlap_doc
    return combo, org_content_mask


  def get_predictions_and_loss(self, input_ids, input_mask, overlap_ids, overlap_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map):
    model = modeling.BertModel(
      config=self.bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      use_one_hot_embeddings=False,
      scope='bert')
    original_doc = model.get_sequence_output()
    model = modeling.BertModel(
      config=self.bert_config,
      is_training=is_training,
      input_ids=overlap_ids,
      input_mask=overlap_mask,
      use_one_hot_embeddings=False,
      scope='bert')
    overlap_doc = model.get_sequence_output()

    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

    num_sentences = tf.shape(input_ids)[0]
    max_sentence_length = tf.shape(input_mask)[1] - 2
    mention_doc, org_content_mask = self.combine_passes(original_doc, input_ids, input_mask, overlap_doc, overlap_ids, overlap_mask)

    num_words = util.shape(mention_doc, 0)
    antecedent_doc = mention_docx
    
    # mask out cross-sentence candidates
    flattened_sentence_indices = sentence_map

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask) # [num_candidates]

    candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

    candidate_span_emb = self.get_span_emb(mention_doc, mention_doc, candidate_starts, candidate_ends) # [num_candidates, emb]
    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

    # beam size
    k = tf.minimum(3900, tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))
    c = tf.minimum(self.config["max_top_antecedents"], k)
    # pull from beam
    top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                               tf.expand_dims(candidate_starts, 0),
                                               tf.expand_dims(candidate_ends, 0),
                                               tf.expand_dims(k, 0),
                                               num_words,
                                               True) # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

    top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
    # don't need this
    top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
    top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
    genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]
    if self.config['use_metadata']:
      speaker_ids = self.flatten_emb_by_sentence(speaker_ids, org_content_mask)
      top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]i
    else:
        top_span_speaker_ids = None


    # antecedent scores -- change this
    dummy_scores = tf.zeros([k, 1]) # [k, 1]
    top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
    num_segs, seg_len = util.shape(org_content_mask, 0), util.shape(org_content_mask, 1)
    word_segments = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, num_segs), 1), [1, seg_len]), [-1])
    flat_word_segments = tf.boolean_mask(word_segments, tf.reshape(org_content_mask, [-1]))
    mention_segments = tf.expand_dims(tf.gather(flat_word_segments, top_span_starts), 1) # [k, 1]
    antecedent_segments = tf.gather(flat_word_segments, tf.gather(top_span_starts, top_antecedents)) #[k, c]
    segment_distance = tf.clip_by_value(mention_segments - antecedent_segments, 0, self.config['max_training_sentences'] - 1) if self.config['use_segment_distance'] else None #[k, c]
    if self.config['fine_grained']:
      for i in range(self.config["coref_depth"]):
        with tf.variable_scope("coref_layer", reuse=(i > 0)):
          top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
          top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb, segment_distance) # [k, c]
          top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
          top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
          attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
          with tf.variable_scope("f"):
            f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1), util.shape(top_span_emb, -1))) # [k, emb]
            top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]
    else:
        top_antecedent_scores = top_fast_antecedent_scores

    top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]

    top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]
    top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask))) # [k, c]
    same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
    non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
    pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
    dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True)) # [k, 1]
    top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]
    loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
    loss = tf.reduce_sum(loss) # []

    return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss


  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
      head_attn_reps = tf.matmul(mention_word_scores, context_outputs) # [K, T]
      span_emb_list.append(head_attn_reps)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]


  def get_mention_scores(self, span_emb, span_starts, span_ends):
      with tf.variable_scope("mention_scores"):
        span_scores = util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]
      if self.config['use_prior']:
        span_width_emb = tf.get_variable("span_width_prior_embeddings", [self.config["max_span_width"], self.config["feature_size"]]) # [W, emb]
        span_width_index = span_ends - span_starts # [NC]
        with tf.variable_scope("width_scores"):
          width_scores =  util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [W, 1]
        width_scores = tf.gather(width_scores, span_width_index)
        span_scores += width_scores
      return span_scores


  def get_width_scores(self, doc, starts, ends):
    distance = ends - starts
    span_start_emb = tf.gather(doc, starts)
    hidden = util.shape(doc, 1)
    with tf.variable_scope('span_width'):
      span_width_emb = tf.gather(tf.get_variable("start_width_embeddings", [self.config["max_span_width"], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02)), distance) # [W, emb]
    scores = tf.reduce_sum(span_start_emb * span_width_emb, axis=1)
    return scores


  def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
      num_words = util.shape(encoded_doc, 0) # T
      num_c = util.shape(span_starts, 0) # NC
      doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1]) # [K, T]
      mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1), doc_range <= tf.expand_dims(span_ends, 1)) #[K, T]
      with tf.variable_scope("mention_word_attn"):
        word_attn = tf.squeeze(util.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)
      mention_word_attn = tf.nn.softmax(tf.log(tf.to_float(mention_mask)) + tf.expand_dims(word_attn, 0))
      return mention_word_attn

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb, segment_distance=None):
    k = util.shape(top_span_emb, 0)
    c = util.shape(top_antecedents, 1)

    feature_emb_list = []

    if self.config["use_metadata"]:
      top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
      same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
      speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), tf.to_int32(same_speaker)) # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)
    if segment_distance is not None:
      with tf.variable_scope('segment_distance', reuse=tf.AUTO_REUSE):
        segment_distance_emb = tf.gather(tf.get_variable("segment_distance_embeddings", [self.config['max_training_sentences'], self.config["feature_size"]]), segment_distance) # [k, emb]
      span_width_emb = tf.nn.dropout(segment_distance_emb, self.dropout)
      feature_emb_list.append(segment_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

    with tf.variable_scope("slow_antecedent_scores"):
      slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
    slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]
    return slow_antecedent_scores # [k, c]

  def get_fast_antecedent_scores(self, top_span_emb):
    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), self.dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())

    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
      
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index, (i, predicted_index)
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted
    
  def get_mention_top_predicted_antecedents(self,top_span_starts, top_span_ends, predicted_antecedents, list_mentions):
    """
    return a dictionary of mapping between
    every mention in a defined list of mentions and its top predicted antecedent
    list_mentions = defined list of mentions (e.g. masked mentions)
    """
    mention_to_predicted_antecedent = {}
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index < 0:
            continue
        assert i > predicted_index, (i,predicted_index)
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        if mention in list_mentions:
            mention_to_predicted_antecedent[mention] = predicted_antecedent
    return mention_to_predicted_antecedent


  def evaluate_antecedent(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, accuracy_evaluator, selected_mentions = None, evaluator_selected = None):
    """
    to update antecedent evaluation and return scores (precision, recall, f1)
    """
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    first_mentions = []
    for gc in gold_clusters:
      if len(gc) > 1:
          first_mention = sorted(gc, key= lambda x: x[0])[0]
          first_mentions.append(first_mention)
          for mention in gc:
            if mention != first_mention:
                mention_to_gold[mention] = gc
    mention_predicted_top_antecedents = self.get_mention_top_predicted_antecedents(top_span_starts, top_span_ends, predicted_antecedents, mention_to_gold.keys())
    accuracy_evaluator.update(mention_predicted_top_antecedents, mention_to_gold)
    if evaluator_selected != None:
        mention_to_gold_selected = {m: mention_to_gold[m] for m in selected_mentions}
        selected_mentions_predicted_top_antecedents = {m: mention_predicted_top_antecedents[m] for m in selected_mentions if m in mention_predicted_top_antecedents}
        evaluator_selected.update(selected_mentions_predicted_top_antecedents, mention_to_gold_selected)
   
  
  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator, selected_mentions = None,
                    evaluator_selected = None):
    '''
    possible to evaluate on selected mentions
    '''
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    if selected_mentions != None:
        evaluator_selected.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold, selected_mentions = selected_mentions)
    return predicted_clusters

  def load_eval_data(self, masked = False):
    '''
    evaluation without masks or only on masks
    '''
    if masked:
        eval_data = self.eval_data_masks
    else:
        eval_data = self.eval_data
        
    if eval_data is None:
      def load_line(line):
        example = json.loads(line)
        #If masked: mask one mention per cluster (but no more than a certain percentage overall)
        if masked:
            example = add_masks(example, skip_first_mention = True, n_masks_profile = 1, all_profiles = False,
                                mask_profile = 'one_per_cluster', n_placeholders = self.n_placeholders, perc_mask = 20)[0]
        return self.tensorize_example(example, is_training=False), example
        
      with open(self.config["eval_path"]) as f:
        eval_data = [load_line(l) for l in f.readlines()]
        
        if masked:
            self.eval_data_masks = eval_data
        else:
            self.eval_data = eval_data
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in eval_data)
      if not masked:
        print("Loaded {} eval examples.".format(len(eval_data)))
      else:
        print("Loaded {} eval examples with masks.".format(len(eval_data)))

  def evaluate(self, session, global_step=None, official_stdout=False, keys=None, eval_mode=False, masked = False, eval_on_masks_only = False, selected_mentions = None):
    '''
    evaluation on all mentions and a subset of them +
    antecedent evaluation
    '''
    
    if masked:
        self.eval_data_masks = None
        
    self.load_eval_data(masked = masked)
    
    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    antecedent_evaluator = metrics.CorefEvaluator_antecedent() # added to make an antecedent_evaluator, parallel to the original evaluator
    if eval_on_masks_only:
        antecedent_evaluator_selected =  metrics.CorefEvaluator_antecedent()
    else:
        antecedent_evaluator_selected = None
    losses = []
    doc_keys = []
    num_evaluated= 0
    
    if masked:
        eval_data = self.eval_data_masks
    else:
        eval_data = self.eval_data
    
    for example_num, (tensorized_example, example) in enumerate(eval_data):
      _, _, _, _, _, _, _, _, gold_starts, gold_ends, _, _ = tensorized_example
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      print(num_evaluated)
      print([i.shape for i in tensorized_example[:4]])
      doc_keys.append(example['doc_key'])
      
      loss, (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores) = session.run([self.loss, self.predictions], feed_dict=feed_dict)
      
      losses.append(loss)
      predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)

      if eval_on_masks_only and masked:
        selected_mentions = example['masked_mentions']
      else:
        selected_mentions = None
            
      coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator) #, selected_mentions = selected_mentions, evaluator_selected = coref_evaluator_selected)
    
      gold_clusters = [tuple(tuple(m) for m in gc) for gc in example['clusters']]
      # added to update the antecedent_evaluation
      antecedent_evaluate_update = self.evaluate_antecedent(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], antecedent_evaluator, selected_mentions = selected_mentions, evaluator_selected  = antecedent_evaluator_selected)
      
      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(eval_data)))

    summary_dict = {}
    if eval_mode:
      conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, self.subtoken_maps, official_stdout )
      average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
      summary_dict["Average F1 (conll)"] = average_f1
      print("Average F1 (conll): {:.2f}%".format(average_f1))
 
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
        return util.make_summary(summary_dict), f, antecedent_f, antecedent_f_masks
    else:
        return util.make_summary(summary_dict), f, antecedent_f

def mentions_to_consider(mentions):
    '''
    We don't mask mentions that are included in other mentions
    '''
    keep = []
    i = 0
    j = i + 1
    while i < len(mentions) and j < len(mentions):
        if mentions[i][1] < mentions[j][0]:
            keep.append(mentions[i])
            i = j
            j += 1
            if j == len(mentions):
                keep.append(mentions[j -1 ])
                break
        else:
            if mentions[i][1] < mentions[j][1]:
                i = j
                j += 1
            else:
                j += 1
    return keep


def build_masks_profiles(candidate_mentions_to_mask, clusters,
                         all_profiles = False, mask_profile = 'percentage', perc_mask = 20, skip_first = False,
                         n_profiles = 5, n_masks_profile = 1, mentions_distance = 5, tokens_distance =50):
    masked_mentions_profile = []
    clusters_tmp = clusters.copy()
    
    if skip_first:
        clusters_tmp = [c for c in clusters_tmp if len(c) > 1]
    
    for i in range(len(clusters_tmp)):
        clusters_tmp[i] = sorted(clusters_tmp[i], key=lambda x: x[0])
        j = 0
        if skip_first:
            while j < len(clusters_tmp[i]):
                if clusters_tmp[i][j] in candidate_mentions_to_mask:
                    clusters_tmp[i] = clusters_tmp[i][j + 1:]
                else:
                    j += 1
        clusters_tmp[i] = [m for m in clusters_tmp[i] if m in candidate_mentions_to_mask]
    
    if len(candidate_mentions_to_mask) > 0:
        if mask_profile == 'percentage':
            n_to_mask = int(round(len(candidate_mentions_to_mask) / 100 * perc_mask, 0))
            if n_to_mask > len(candidate_mentions_to_mask):
                n_to_mask = len(candidate_mentions_to_mask)
            random.shuffle(candidate_mentions_to_mask)
            if n_to_mask > 0:
                while candidate_mentions_to_mask != []:
                    # mask a % of mentions in the textspan
                    masked_mentions = []
                    for _ in range(n_to_mask):
                        masked_mentions.append(candidate_mentions_to_mask.pop())
                        if candidate_mentions_to_mask == []:
                            break
                    masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                    masked_mentions_profile.append(masked_mentions)
                    if not all_profiles and len(masked_mentions_profile) == n_masks_profile:
                        break
        elif mask_profile == 'one_per_cluster':
            n_to_mask = int(round(len(candidate_mentions_to_mask) / 100 * perc_mask, 0))
            random.shuffle(clusters_tmp)
            for i in range(len(clusters_tmp)):
                random.shuffle(clusters_tmp[i])
            while candidate_mentions_to_mask != []:
                masked_mentions = []
                for i in range(len(clusters_tmp)):
                    if len(clusters_tmp[i]) > 0:
                        to_mask = clusters_tmp[i].pop()
                        masked_mentions.append(to_mask)
                        candidate_mentions_to_mask.remove(to_mask)
                        if len(masked_mentions) == n_to_mask:
                            break
                masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                masked_mentions_profile.append(masked_mentions)
                if not all_profiles and len(masked_mentions_profile) == n_masks_profile:
                       break
        elif mask_profile == 'n_profiles_random':
            if len(candidate_mentions_to_mask) >= n_profiles:
                random.shuffle(candidate_mentions_to_mask)
                n = len(candidate_mentions_to_mask)//n_profiles
                for i in range(n_profiles):
                    masked_mentions = []
                    if i == n_profiles -1 :
                        n += len(candidate_mentions_to_mask) % n_profiles
                    for _ in range(n):
                        masked_mentions.append(candidate_mentions_to_mask.pop())
                        if candidate_mentions_to_mask == []:
                            break
                    masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                    masked_mentions_profile.append(masked_mentions)
        elif mask_profile == 'n_profiles_ordered':
            if len(candidate_mentions_to_mask) >= n_profiles:
                candidate_mentions_to_mask = sorted(candidate_mentions_to_mask, key=lambda x: x[0])
                for i in range(n_profiles):
                    masked_mentions = []
                    j = i
                    while j < len(candidate_mentions_to_mask):
                        masked_mentions.append(candidate_mentions_to_mask[j])
                        j += n_profiles
                    masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                    masked_mentions_profile.append(masked_mentions)
        elif mask_profile == 'len_profiles_random':
            if n_masks_profile <= len(candidate_mentions_to_mask):
                while candidate_mentions_to_mask != []:
                    # mask a % of mentions in the textspan
                    masked_mentions = []
                    for _ in range(n_masks_profile):
                        masked_mentions.append(candidate_mentions_to_mask.pop())
                        if candidate_mentions_to_mask == []:
                            break
                    masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                    masked_mentions_profile.append(masked_mentions)
                    if not all_profiles and len(masked_mentions_profile) == n_masks_profile:
                        break
        elif mask_profile == 'len_profiles_ordered':
            if n_masks_profile <= len(candidate_mentions_to_mask):
                candidate_mentions_to_mask = sorted(candidate_mentions_to_mask, key=lambda x: x[0])
                n = len(candidate_mentions_to_mask)//n_masks_profile + 1
                i = 0
                added = []
                while sum([len(m) for m in masked_mentions_profile]) != len(candidate_mentions_to_mask):
                    masked_mentions = []
                    j = i
                    while j < len(candidate_mentions_to_mask):
                        if candidate_mentions_to_mask[j] not in added:
                            masked_mentions.append(candidate_mentions_to_mask[j])
                            added.append(candidate_mentions_to_mask[j])
                        j += n
                    masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                    masked_mentions_profile.append(masked_mentions)
                    i += 1
                    if not all_profiles and len(masked_mentions_profile) == n_masks_profile:
                        break
        elif mask_profile == 'mentions_distance':
            if n_masks_profile <= len(candidate_mentions_to_mask):
                candidate_mentions_to_mask = sorted(candidate_mentions_to_mask, key=lambda x: x[0])
                i = 0
                added = []
                while sum([len(m) for m in masked_mentions_profile]) != len(candidate_mentions_to_mask):
                    masked_mentions = []
                    j = i
                    while j < len(candidate_mentions_to_mask):
                        if candidate_mentions_to_mask[j] not in added:
                            masked_mentions.append(candidate_mentions_to_mask[j])
                            added.append(candidate_mentions_to_mask[j])
                        j += mentions_distance
                    masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                    masked_mentions_profile.append(masked_mentions)
                    i += 1
            random.shuffle(masked_mentions_profile)
            if not all_profiles:
                masked_mentions_profile = masked_mentions_profile[:n_masks_profile]
        elif mask_profile == 'tokens_distance':
            candidate_mentions_to_mask = sorted(candidate_mentions_to_mask, key=lambda x: x[0])
            i = 0
            added = []
            while sum([len(m) for m in masked_mentions_profile]) != len(candidate_mentions_to_mask):
                masked_mentions = []
                j = i
                while j < len(candidate_mentions_to_mask):
                    if candidate_mentions_to_mask[j] not in added:
                        if not masked_mentions == []:
                            if candidate_mentions_to_mask[j][0] >= masked_mentions[-1][1] + tokens_distance:
                                masked_mentions.append(candidate_mentions_to_mask[j])
                                added.append(candidate_mentions_to_mask[j])
                        else:
                            masked_mentions.append(candidate_mentions_to_mask[j])
                            added.append(candidate_mentions_to_mask[j])
                    j += 1
                masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                masked_mentions_profile.append(masked_mentions)
                i += 1
            random.shuffle(masked_mentions_profile)
            if not all_profiles:
                masked_mentions_profile= masked_mentions_profile[:n_masks_profile]
        elif mask_profile == 'one_per_cluster_tokens_distance':
            n_to_mask = int(round(len(candidate_mentions_to_mask) / 100 * perc_mask, 0))
            mention_to_cluster = {}
            for c in clusters_tmp:
                for m in c:
                    mention_to_cluster[tuple(m)] = c
            candidate_mentions_to_mask = sorted(candidate_mentions_to_mask, key=lambda x: x[0])
            i = 0
            added = []
            while sum([len(m) for m in masked_mentions_profile]) != len(candidate_mentions_to_mask):
                masked_mentions = []
                j = i
                while j < len(candidate_mentions_to_mask):
                    if candidate_mentions_to_mask[j] not in added:
                        if not masked_mentions == []:
                            if candidate_mentions_to_mask[j][0] >= masked_mentions[-1][1] + tokens_distance:
                                same_cluster = False
                                for mask in masked_mentions:
                                    if candidate_mentions_to_mask[j] in mention_to_cluster[tuple(mask)]:
                                        same_cluster = True
                                        break
                                if not same_cluster:
                                    masked_mentions.append(candidate_mentions_to_mask[j])
                                    added.append(candidate_mentions_to_mask[j])
                        else:
                            masked_mentions.append(candidate_mentions_to_mask[j])
                            added.append(candidate_mentions_to_mask[j])
                    if len(masked_mentions) == n_to_mask:
                        break
                    j += 1
                masked_mentions = sorted(masked_mentions, key=lambda x: x[0])
                masked_mentions_profile.append(masked_mentions)
                i += 1
            random.shuffle(masked_mentions_profile)
            if not all_profiles:
                masked_mentions_profile = masked_mentions_profile[:n_masks_profile]
    if masked_mentions_profile == []:
        masked_mentions_profile = [[]]
    return masked_mentions_profile
    
    
def add_masks(t, perc_mask=20, mask_profile = 'percentage', max_to_mask = None, n_placeholders = 1,
             all_profiles = False, skip_first_mention = True, tokens_distance = 30, n_masks_profile = 1):
    # get boundaries of textspans and full text
    text_span_all = []
    boundaries_sentences = []
    for i in range(len(t['sentences'])):
        text_span = t['sentences'][i]
        boundaries_sentences.append([len(text_span_all), len(text_span_all) + len(text_span)])
        text_span_all += text_span
    # collect all mentions
    mentions = []
    actual_mentions = {}    # only for debugging
    for cluster in t['clusters']:
        for mention in cluster:
            mention_start = mention[0]
            mentions.append(mention)
            # only for debugging
            actual_mentions[str(mention)] = text_span_all[mention_start:mention[-1] + 1]
    mentions = sorted(mentions, key=lambda x: x[0])

    map_mentions_to_new_indices = {}    # collect indices of mentions shifted after adding masks

    
    candidate_mentions_to_mask = mentions_to_consider(
            mentions)    # keep only mentions that do not include other mentions
    
      
    masked_mentions_profiles = build_masks_profiles(candidate_mentions_to_mask, t['clusters'],
                         all_profiles = all_profiles, mask_profile = mask_profile, perc_mask = perc_mask, skip_first = skip_first_mention,
                         tokens_distance = tokens_distance, n_masks_profile = n_masks_profile)
    
    new_datapoints = []
    
    
    for masked_mentions in masked_mentions_profiles:

        if masked_mentions != []:
            mask_mention_starts = [m[0] for m in masked_mentions]
        else:
            mask_mention_starts = []

        actual_masked_mentions = [actual_mentions[str(m)] for m in masked_mentions]

        token_index = 0
        new_token_index = 0
        prev_subtoken_index = 0    # used for subtoken map
        count_subtokens = 0

        mentions_covered = 0
        idx_next_mention = 0

        speakers_new_all = []    # eventually list of lists
        sentences_new_all = []    # eventually list of lists
        sentence_map_new = []
        subtoken_map_new = []
        pronouns_new = []

        text_span_all_new = []    # for debugging


        for textspan_idx in range(len(t['sentences'])):
            text_span = t['sentences'][textspan_idx]
            speakers = t['speakers'][textspan_idx]
            text_span_new = []
            speakers_new = []
            boundaries_textspan = boundaries_sentences[textspan_idx]    # boundaries of tokens spanned in the larger text

            # find mentions in the textspan
            mentions_textspan = []
            while mentions_covered < len(mentions) and mentions[mentions_covered][1] <= boundaries_textspan[1]:
                mentions_textspan.append(mentions[mentions_covered])
                mentions_covered += 1

            token_index_in_textspan = 0

            while token_index_in_textspan < len(text_span) and token_index < len(
                            text_span_all):    # while considering tokens in the textspan
                subtoken_index = t['subtoken_map'][token_index]
                if token_index in t['pronouns']:
                    pronouns_new.append(new_token_index)
                masked = False
                if idx_next_mention < len(mentions) and token_index == mentions[idx_next_mention][0]:
                    mention = mentions[idx_next_mention]
                    if mention[0] in mask_mention_starts and not mention in masked_mentions:
                        idx_next_mention += 1
                        continue
                    if mention in masked_mentions:
                        prev_subtoken_index = subtoken_index
                        count_subtokens += 1
                        text_span_new.append('[MASK]')
                        speakers_new.append(speakers[token_index_in_textspan])
                        sentence_map_new.append(t['sentence_map'][token_index])
                        subtoken_map_new.append(count_subtokens)
                        map_mentions_to_new_indices[str(mention)] = [new_token_index, new_token_index]
                        new_token_index += 1
                        masked = True
                        current_idx_mention = idx_next_mention
                    else:
                        map_mentions_to_new_indices[str(mention)] = [new_token_index,new_token_index + mention[1] - mention[0]]
                    idx_next_mention += 1
                    if idx_next_mention < len(mentions):
                        if not masked and token_index == mentions[idx_next_mention][0]:
                            continue
                        elif masked:
                            while mentions[idx_next_mention][1] <= mentions[current_idx_mention][1]:
                                idx_next_mention += 1
                if not token_index in mask_mention_starts:
                    if subtoken_index != prev_subtoken_index:
                        count_subtokens += 1
                        prev_subtoken_index = subtoken_index
                    text_span_new.append(text_span[token_index_in_textspan])
                    speakers_new.append(speakers[token_index_in_textspan])
                    sentence_map_new.append(t['sentence_map'][token_index])
                    subtoken_map_new.append(count_subtokens)
                    new_token_index += 1
                    token_index += 1
                    token_index_in_textspan += 1
                else:
                    token_index += mention[1] - mention[0] + 1
                    token_index_in_textspan += mention[1] - mention[0] + 1
            sentences_new_all.append(text_span_new)
            speakers_new_all.append(speakers_new)
            text_span_all_new += text_span_new
        clusters_new = []

        for c in t['clusters']:
            cluster = []
            for m in c:
                if str(m) in map_mentions_to_new_indices.keys():
                    cluster.append(map_mentions_to_new_indices[str(m)])
            if cluster != []:
                clusters_new.append(cluster)
        t_new = t.copy()

        try:
            new_masked_mentions = [tuple(map_mentions_to_new_indices[str(m)]) for m in masked_mentions]
        except:
            print(masked_mentions, map_mentions_to_new_indices)
            print(t)
        t_new['masked_mentions'] = new_masked_mentions
        t_new['masked_mentions_in_original'] = [tuple(m) for m in masked_mentions]
        t_new['clusters'] = clusters_new
        t_new['sentences'] = sentences_new_all
        t_new['sentence_map'] = sentence_map_new
        t_new['pronouns'] = pronouns_new
        t_new['speakers'] = speakers_new_all
        t_new['subtoken_map'] = subtoken_map_new
        new_datapoints.append(t_new)
    return new_datapoints
