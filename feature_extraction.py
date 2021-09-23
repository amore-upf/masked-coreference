#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 11:33:26 2021

@author: liaoxixian
"""
import OntoNotes_AllenNLP as onto
import os
import numpy as np
from scipy.stats import entropy 
import itertools 
import operator
                                                                                                                                                                      
                                                              
class features(object):
    
    def __init__(self, dataset_path, example, target_mentions):
        self.list_path = self.dataset_path_list(dataset_path)
        self.example = example
        self.target_mentions = target_mentions       
                                                             
    def dataset_path_list(self, dataset_path: str):
        """
        A list containing file_paths in a directory
        containing CONLL-formatted files.
        """
        list_path = []
        for root, _, files in list(os.walk(dataset_path)):
            for data_file in files:
                if not data_file.endswith("gold_conll"):
                    continue
                list_path.append(os.path.join(root, data_file))
        return list_path
        

    
    def key_to_path_and_sentence_index (self): 
        input_doc_key = self.example['doc_key']
        key_info = input_doc_key.split('/')
        file_name =  ('_').join(key_info[-1].split('_')[:2]) + str('.gold_conll')   
        sentence_index = key_info[-1].split('_')[-1]
        key = key_info[:3]
        key.append(file_name) 
        for p in self.list_path:
            p_split = p.split('/')
            if key == p_split[-4:]:
                doc_path = p
                return doc_path, sentence_index

            
    def mention_to_genre (self):
        "the genre of the text where the mention is"
        gold_mentions = self.gold_mentions()
        mention_to_genre = {}
        input_doc_key = self.example['doc_key']
        genre = input_doc_key.split('/')[0]
        for m in gold_mentions:
            mention_to_genre[m] = genre
        return mention_to_genre
            
                   
    
    def where_sentences_in_ontonotes(self):
        try:
            doc_path, sentence_index = self.key_to_path_and_sentence_index()
            ON= onto.Ontonotes()
            sentences = ON.sentence_iterator(file_path= doc_path)
            list_sentences = []
            for s in sentences:
                if s.sentence_id == int(sentence_index):
                    list_sentences.append(s)
            return list_sentences

        except TypeError:
            print('fail to locate example in OntoNotes:', self.example)  
     
           

    def gold_mentions(self):
        gold_mentions =[] 
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example['clusters']] 
        for gc in gold_clusters:
            for m in gc:
                gold_mentions.append(m)
        return gold_mentions
    

    def mention_to_sentence (self): 
        gold_mentions = self.gold_mentions()
        mention_to_sentence_map = {}
        sentence_map = self.example['sentence_map']
        for m in gold_mentions:
            mention_to_sentence_map[m] = [sentence_map[m[0]]]
        return mention_to_sentence_map 


    def mention_to_sentence_subtoken (self):      
        if 'masked_mentions' in self.example:      
            from_original_to_new_indices = {tuple(eval(k)): tuple(v) for k,v in  self.example['from_original_to_new_indices'].items()}
            original_mentions = list(from_original_to_new_indices.keys())
            new_mention_to_sentence_map = self.mention_to_sentence()
            list_sentences = self.where_sentences_in_ontonotes()
            original_subtoken_map = self.example['original_subtoken_map']
            original_mention_original_subtoken_map = {}
            for m in original_mentions:
                original_mention_original_subtoken_map[m] = (original_subtoken_map[m[0]], original_subtoken_map[m[1]])
            range_sents = [0]
            index = 0
            for s in list_sentences:
                index += len(s.words)
                range_sents.append(index)
            span_sents = [(range_sents[i], range_sents[i + 1]-1) for i in range(len(range_sents) - 1)]            
            for i,span in enumerate(span_sents):
                for m in original_mentions:
                    if original_mention_original_subtoken_map[m][0] >= span[0] and\
                    original_mention_original_subtoken_map[m][1] <=  span[1]:
                        if i == 0: 
                            new_subtoken = original_mention_original_subtoken_map[m]
                            new_mention = from_original_to_new_indices[m]
                            new_mention_to_sentence_map[new_mention].append(new_subtoken)
                        else:
                            new_subtoken = (original_mention_original_subtoken_map[m][0]-span_sents[i-1][1]-1, original_mention_original_subtoken_map[m][1]-span_sents[i-1][1]-1)
                            new_mention = from_original_to_new_indices[m]
                            new_mention_to_sentence_map[new_mention].append(new_subtoken)
            new_mention_original_sentence_subtoken_map = new_mention_to_sentence_map
            return new_mention_original_sentence_subtoken_map 
        else:
            gold_mentions = self.gold_mentions()
            mention_to_sentence_map = self.mention_to_sentence()
            list_sentences = self.where_sentences_in_ontonotes()
            subtoken_map = self.example['subtoken_map']
            mention_subtoken_map = {}
            for m in gold_mentions:
                mention_subtoken_map[m] = (subtoken_map[m[0]], subtoken_map[m[1]])
            range_sents = [0]
            index = 0
            for s in list_sentences:
                index += len(s.words)
                range_sents.append(index)
            span_sents = [(range_sents[i], range_sents[i + 1]-1) for i in range(len(range_sents) - 1)]            
            for i,span in enumerate(span_sents):
                for m in gold_mentions:
                    if mention_subtoken_map[m][0] >= span[0] and\
                    mention_subtoken_map[m][1] <=  span[1]:
                        if i == 0: 
                            new_subtoken = mention_subtoken_map[m]
                            mention_to_sentence_map[m].append(new_subtoken)
                        else:
                            new_subtoken = (mention_subtoken_map[m][0]-span_sents[i-1][1]-1, mention_subtoken_map[m][1]-span_sents[i-1][1]-1)
                            mention_to_sentence_map[m].append(new_subtoken)
            mention_sentence_subtoken_map = mention_to_sentence_map
            return  mention_sentence_subtoken_map
        
        
       
      
         
    def mention_masked_or_not (self):
        mention_masked_or_unmasked = {}
        target_mentions = self.target_mentions
        masked_mentions = self.example['masked_mentions']
        for m in target_mentions:
            if m in masked_mentions:
                mention_masked_or_unmasked[m] = 'masked'
            else:
                mention_masked_or_unmasked[m] = 'unmasked'
        mention_masked_or_unmasked = {str(k): v for k, v in mention_masked_or_unmasked.items()}                
        return mention_masked_or_unmasked
        
    
    def new_to_original_indices (self): 
        "mapping between original indices and new indices after masking mentions"  
        
        original_to_new = self.example['from_original_to_new_indices']
        new_to_original = {str(tuple(v)): tuple(eval(k)) for k, v in original_to_new.items()}
        new_to_original['none'] = 'no_antecedent'
        return new_to_original
    
  
    def mention_last_antecedent(self):
       "mapping between mentions and their cloest antecedent"
       "the antecedent for first mentions is 'none'"
       
       mention_recent_antecedent = {}
       gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example['clusters']] 
       descending_clusters = []
       for gc in gold_clusters:
           new = sorted(gc,key=lambda x:(-x[0],-x[1]))   
           descending_clusters.append(new)
       for reordered_gc in descending_clusters:
           for i,m in enumerate(reordered_gc):
                if m in self.target_mentions:
                   if i == len(reordered_gc)-1:
                       mention_recent_antecedent[m] =  'none'
                   else:
                       mention_recent_antecedent[m] =  reordered_gc[i+1] 
       return mention_recent_antecedent
   

    def mention_to_words_pos (self):
        "part-of-speech tags of mentions"
        
        mention_sentence_subtoken_map = self.mention_to_sentence_subtoken() 
        list_sentences = self.where_sentences_in_ontonotes()
        mention_words_pos = {}
        for m in mention_sentence_subtoken_map:
            sentence_index = mention_sentence_subtoken_map[m][0]
            token_index= mention_sentence_subtoken_map[m][1]
            sentence_pos = list_sentences[sentence_index].pos_tags
            sentence_words = list_sentences[sentence_index].words
            m_pos = sentence_pos[token_index[0]:token_index[1]+1]
            m_words = sentence_words[token_index[0]:token_index[1]+1]
            mention_words_pos[m] = [m_words, m_pos]
        return mention_words_pos     

    
    def mention_to_string_mention (self):
        "mention indices mapped to strings of mention"
        
        mention_to_string = {}
        mention_words_pos = self.mention_to_words_pos () 
        def list_of_str_to_str (list_of_str):
            mention_string = ' '.join(list_of_str)
            return mention_string
        for m in  mention_words_pos:
            mention_to_string[m] = list_of_str_to_str (mention_words_pos[m][0])
        mention_to_string = {str(k): v for k, v in mention_to_string.items()}
        return mention_to_string

   
    def last_antecedent_to_string (self):
        "indices of the closest antecedent mapped to its strings"
        
        antecedent_to_string = {}
        mention_recent_antecedent = self.mention_last_antecedent()
        mention_words_pos = self.mention_to_words_pos ()
        def list_of_str_to_str (list_of_str):
            mention_string = ' '.join(list_of_str)
            return mention_string        
        for m in mention_recent_antecedent: 
            if mention_recent_antecedent[m] == 'none':
                antecedent_to_string[m] = 'no_antecedent'
            else:
                antecedent_words_pos = mention_words_pos[mention_recent_antecedent[m]]        
                antecedent_to_string[m] = list_of_str_to_str (antecedent_words_pos[0])
        return antecedent_to_string        
  
            
    
    def mention_to_mentiontype (self):
        "mapping between mention and its type"
        
        mention_words_pos = self.mention_to_words_pos ()
        mention_type = {}
        for m in mention_words_pos:            
            if mention_words_pos[m][1] == ['PRP']:
                # pronoun
                if mention_words_pos[m][0][0] in ['I', 'i', 'me', 'Me', 'We', 'we', 'Us','us', 'myself','Myself', 'ourselves', 'Ourselves']:
                    mention_type[m] = 'first_person_pronoun'
                elif mention_words_pos[m][0][0] in ['You', 'you', 'yourself', 'Yourself', 'Yourselves', 'youselves']:
                    mention_type[m] = 'second_person_pronoun'
                elif mention_words_pos[m][0][0] in ['He','he','himself', 'Himself', 'She','she', 'herself', 'Herself','They', 'they', 'themselves', 'Themselves', 'It', 'it', 'itself', 'Itself', 'Him', 'him', 'Her', 'her', 'Them', 'them']:
                    mention_type[m] = 'third_person_pronoun'
                else:
                    mention_type[m] = 'other_pronoun'
            elif len(mention_words_pos[m][1]) == 1 and mention_words_pos[m][0][0] in ['This','this', 'That', 'that', 'These','these', 'Those','those']:
                mention_type[m] = 'demonstrative_pronoun'                
            elif mention_words_pos[m][1] == ['PRP$']:
                if mention_words_pos[m][0][0] in ['mine', 'Mine', 'ours', 'Ours','my','My','Our', 'our']:
                    mention_type[m] = 'first_person_possessive_pronoun'
                elif mention_words_pos[m][0][0] in ['yours', 'Yours', 'Your','your']:
                    mention_type[m] = 'second_person_possessive_pronoun'
                elif mention_words_pos[m][0][0] in ['his', 'His', 'hers', 'Hers', 'its', 'Its', 'theirs', 'Theirs', 'Her','her', 'their', 'Their']:
                    mention_type[m] = 'third_person_possessive_pronoun'
                else:
                    mention_type[m] = 'other_possessive_pronoun' 
            elif set(mention_words_pos[m][1]) in [{'NNP'} ,{'NNP','HYPH'}]:
                mention_type[m] = 'proper_noun' 
            elif ',' in mention_words_pos[m][1] and set(mention_words_pos[m][1][:mention_words_pos[m][1].index(',')]) in [{'NNP'}, {'NNP', 'NN'}]:
                # proper noun phrase + sth else (e.g. appositive clauses)
                mention_type[m] = 'proper_noun_appositive_clause'
            elif mention_words_pos[m][1][0] == 'DT' and  mention_words_pos[m][0][0] in ['the','The', 'this', 'This', 'that', 'That','These','these','those','Those']:
                mention_type[m] = 'definite_NP'                
            elif len(mention_words_pos[m][1]) > 1 and mention_words_pos[m][1][0] == 'PRP$':
                #possessive pronouns + nouns
                mention_type[m] = 'possessive_pronoun_NP'               
            elif mention_words_pos[m][1][0] == 'DT' and  mention_words_pos[m][0][0] not in ['the','The', 'this', 'This', 'that', 'That','These','these','those','Those']:
                mention_type[m] = 'indefinite_NP'
            elif set(mention_words_pos[m][1]) in [{'NNS'}, {'NN'}, {'NNS','NN'}, {'NNP', 'NN'}, {'NNP','NNS'}]:
                #generic noun e.g. children
                mention_type[m] = 'indefinite_NP'               
            elif len(set(mention_words_pos[m][1])) in [2,3] and ('NNP' and 'POS') in set(mention_words_pos[m][1]):
                #possessive phrase: Fitty Cent's
                mention_type[m] = 'possessive_phrase' 
            elif 'WP$' in mention_words_pos[m][1] and set(mention_words_pos[m][1][:mention_words_pos[m][1].index('WP$')]) in [{'NNP'}, {'NNP', 'NN'}]:
                # proper noun phrase + sth else (e.g. relative clause)
                mention_type[m] = 'proper_noun_relative_clause'
            elif 'WRB' in mention_words_pos[m][1] and set(mention_words_pos[m][1][:mention_words_pos[m][1].index('WRB')]) in [{'NNP'}, {'NNP', 'NN'}]:
                # proper noun phrase + sth else (e.g. where relative clause) 
                mention_type[m] = 'proper_noun_relative_clause'
                
            elif mention_words_pos[m][1][0] == 'CD' and set(mention_words_pos[m][1][1:]) in [{'NN'},{'NNS'},{'NN','NNS'}]:
                # noun phrases containing numeral quantifiers 
                mention_type[m] = 'cardinal_NP'
            elif mention_words_pos[m][1][0] == 'JJ' and  mention_words_pos[m][0][0] in ['many','Many','some','Some']:
                mention_type[m] = 'indefinite_NP'                

            else: 
                mention_type[m] = 'other'              
        mention_type = {str(k): v for k, v in mention_type.items()}
        return mention_type
     


    def distance_to_last_antecedent (self):        
       "the number of tokens between a mention and its gold closest antecedent"  
       
       mention_recent_antecedent = self.mention_last_antecedent()
       subtoken_map = self.example['subtoken_map']
       mention_antecedent_subtoken_map = {}
       for m in mention_recent_antecedent:
           if mention_recent_antecedent[m] == 'none':
               mention_subtoken = (subtoken_map[m[0]], subtoken_map[m[1]])
               mention_antecedent_subtoken_map[mention_subtoken] = 'no_antecedent'
           else:
               mention_subtoken = (subtoken_map[m[0]], subtoken_map[m[1]])
               antecedent_subtoken = (subtoken_map[mention_recent_antecedent[m][0]], subtoken_map[mention_recent_antecedent[m][1]])
               mention_antecedent_subtoken_map[mention_subtoken] =  antecedent_subtoken
       mention_last_antecedent_distance = {}
       for i,m in enumerate(mention_antecedent_subtoken_map):
           mention_in_example =  list(mention_recent_antecedent.keys())[i]
           if mention_antecedent_subtoken_map[m] == 'no_antecedent':
               mention_last_antecedent_distance[mention_in_example] = 'no_antecedent' #the very frist mention of an entity
           else:
               current_mention_start = m[0]
               antecedent_end = mention_antecedent_subtoken_map[m][1]
               mention_last_antecedent_distance[mention_in_example] = current_mention_start - antecedent_end
       mention_last_antecedent_distance = {str(k): v for k, v in mention_last_antecedent_distance.items()}
       return mention_last_antecedent_distance     
     

    def antecedent_type (self):
        "type of the gold closest antecedent"
        
        mention_words_pos = self.mention_to_words_pos ()
        mention_recent_antecedent = self.mention_last_antecedent()
        antecedent_mentiontype = {}
        for m in mention_recent_antecedent: 
            if mention_recent_antecedent[m] == 'none':
                antecedent_mentiontype[m] = 'no_antecedent'
            else:
                antecedent_words_pos = mention_words_pos[mention_recent_antecedent[m]]
                if antecedent_words_pos[1] == ['PRP']:
                    if antecedent_words_pos[0][0] in ['I', 'i', 'me', 'Me', 'We', 'we', 'Us','us', 'myself','Myself', 'ourselves', 'Ourselves']:
                        antecedent_mentiontype[m] = 'first_person_pronoun'
                    elif antecedent_words_pos[0][0] in ['You', 'you', 'yourself', 'Yourself', 'Yourselves', 'youselves']:
                        antecedent_mentiontype[m] = 'second_person_pronoun'
                    elif antecedent_words_pos[0][0] in ['He','he','himself', 'Himself', 'She','she', 'herself', 'Herself','They', 'they', 'themselves', 'Themselves', 'It', 'it', 'itself', 'Itself', 'Him', 'him', 'Her', 'her', 'Them', 'them']:
                        antecedent_mentiontype[m] = 'third_person_pronoun'
                    else:
                        antecedent_mentiontype[m] = 'other_pronoun'
                elif len(antecedent_words_pos[1]) == 1 and antecedent_words_pos[0][0] in ['This','this', 'That', 'that', 'These','these', 'Those','those']:
                    antecedent_mentiontype[m] = 'demonstrative_pronoun'     
                elif antecedent_words_pos[1] == ['PRP$']:
                    if antecedent_words_pos[0][0] in ['mine', 'Mine', 'ours', 'Ours','my','My','Our', 'our']:
                        antecedent_mentiontype[m] = 'first_person_possessive_pronoun'
                    elif antecedent_words_pos[0][0] in ['yours', 'Yours', 'Your','your']:
                        antecedent_mentiontype[m] = 'second_person_possessive_pronoun'
                    elif antecedent_words_pos[0][0] in ['his', 'His', 'hers', 'Hers', 'its', 'Its', 'theirs', 'Theirs', 'Her','her', 'their', 'Their']:
                        antecedent_mentiontype[m] = 'third_person_possessive_pronoun'
                    else:
                        antecedent_mentiontype[m] = 'other_possessive_pronoun'                 
                elif set(antecedent_words_pos[1]) in [{'NNP'} ,{'NNP','HYPH'}] :
                    antecedent_mentiontype[m] = 'proper_noun' 
                elif ',' in antecedent_words_pos[1] and set(antecedent_words_pos[1][:antecedent_words_pos[1].index(',')]) in [{'NNP'}, {'NNP', 'NN'}]:
                    # proper noun phrase + sth else (e.g. appositive clauses)
                    antecedent_mentiontype[m] = 'proper_noun_appositive_clause'
                elif antecedent_words_pos[1][0] == 'DT' and  antecedent_words_pos[0][0] in ['the','The', 'this', 'This', 'that', 'That','These','these','those','Those']:
                    antecedent_mentiontype[m] = 'definite_NP'                
                elif len(antecedent_words_pos[1]) > 1 and antecedent_words_pos[1][0] == 'PRP$':
                    #possessive pronouns + nouns
                    antecedent_mentiontype[m] = 'possessive_pronoun_NP'               
                elif antecedent_words_pos[1][0] == 'DT' and  antecedent_words_pos[0][0] not in ['the','The', 'this', 'This', 'that', 'That','These','these','those','Those']:
                    antecedent_mentiontype[m] = 'indefinite_NP'
                elif set(antecedent_words_pos[1]) in [{'NNS'}, {'NN'}, {'NNS','NN'}, {'NNP', 'NN'}, {'NNP','NNS'}]:
                    #generic noun e.g. children
                    antecedent_mentiontype[m] = 'indefinite_NP'               
                elif len(set(antecedent_words_pos[1])) in [2,3]  and ('NNP' and 'POS') in set(antecedent_words_pos[1]):
                    #possessive phrase: Fitty Cent's
                    antecedent_mentiontype[m] = 'possessive_phrase' 
                elif 'WP$' in antecedent_words_pos[1] and set(antecedent_words_pos[1][:antecedent_words_pos[1].index('WP$')]) in [{'NNP'}, {'NNP', 'NN'}]:
                    # proper noun phrase + sth else (e.g. relative clause)
                    antecedent_mentiontype[m] = 'proper_noun_relative_clause'
                elif 'WRB' in antecedent_words_pos[1] and set(antecedent_words_pos[1][:antecedent_words_pos[1].index('WRB')]) in [{'NNP'}, {'NNP', 'NN'}]:
                    # proper noun phrase + sth else (e.g. where relative clause) 
                    antecedent_mentiontype[m] = 'proper_noun_relative_clause'

                elif antecedent_words_pos[1][0] == 'CD' and set(antecedent_words_pos[1][1:]) in [{'NN'},{'NNS'},{'NN','NNS'}]:
                    # noun phrases containing numeral quantifiers 
                    antecedent_mentiontype[m] = 'cardinal_NP'
                elif antecedent_words_pos[1][0] == 'JJ' and  antecedent_words_pos[0][0] in ['many','Many','some','Some']:
                    antecedent_mentiontype[m] = 'indefinite_NP'                

                else: 
                    antecedent_mentiontype[m] = 'other'                                             
        antecedent_mentiontype = {str(k): v for k, v in antecedent_mentiontype.items()}
        return antecedent_mentiontype    

    
    def number_of_distractors (self):
        "number of intervening mentions between a mention and its gold closest antecedent"
        # first mention --> 'no_antecedent'  
        # 0 --> the closest antecedent is the very last mention before the current mention 
        
        mention_number_distractors = {}
        gold_mentions = self.gold_mentions()
        reordered_mentions = sorted(gold_mentions,key=lambda x:(-x[0],-x[1])) 
        mention_recent_antecedent = self.mention_last_antecedent()
        for m in self.target_mentions:
            if mention_recent_antecedent[m] == 'none':
                mention_number_distractors[m] = 'no_antecedent'  # first mention
            else:
                m_index = reordered_mentions.index(m)
                antecedent_index = reordered_mentions.index(mention_recent_antecedent[m])
                mention_number_distractors[m] = antecedent_index - m_index - 1
        mention_number_distractors = {str(k): v for k, v in mention_number_distractors.items()}
        return mention_number_distractors


    def distractive_entities (self):
        "number of intervening entities between a mention and its gold closest antecedent"
        # no_antecedent --> the current mention is the first mention of an entity  
        # 0             --> the closest antecedent is the very last mention before the current mention 
        
        mention_distractive_entities = {}
        gold_mentions = self.gold_mentions()
        reordered_mentions = sorted(gold_mentions,key=lambda x:(-x[0],-x[1])) 
        mention_recent_antecedent = self.mention_last_antecedent()
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example['clusters']] 
        for m in mention_recent_antecedent:
            if mention_recent_antecedent[m] == 'none':
                mention_distractive_entities[m] = 'no_antecedent'
            else:
                m_index = reordered_mentions.index(m)
                antecedent_index = reordered_mentions.index(mention_recent_antecedent[m])
                distractors = reordered_mentions[m_index+1:antecedent_index]

                number_distractive_entities = 0
                for cluster in gold_clusters:
                    if len(list(set(distractors).intersection(cluster))) != 0:
                        number_distractive_entities += 1  
                mention_distractive_entities[m] = number_distractive_entities    
        mention_distractive_entities = {str(k): v for k, v in mention_distractive_entities.items()}
        return mention_distractive_entities

    
       
    def mention_previous_tokens (self):
        "number of tokens in the previous context"
        
        mention_to_previous_tokens = {}
        target_mentions = self.target_mentions
        for m in target_mentions:
            mention_to_previous_tokens[m] = m[0]
        return mention_to_previous_tokens
        
        

    def mention_previous_mentions (self):
        "list of all the mentions in the previous context"
        
        previous_mentions = {}
        gold_mentions = self.gold_mentions()
        target_mentions = self.target_mentions
        reordered_mentions = sorted(gold_mentions,key=lambda x:(x[0],x[1])) 
        for m in target_mentions:
            m_index = reordered_mentions.index(m)
            previous_mentions[m] = reordered_mentions[:m_index]    
        return previous_mentions   

       

    def number_of_previous_mentions (self):        
        "number of mentions in the previous context"
        
        mention_number_previous_mentions = {}
        gold_mentions = self.gold_mentions()
        target_mentions = self.target_mentions
        reordered_mentions = sorted(gold_mentions,key=lambda x:(x[0],x[1])) 
        for m in target_mentions:
            m_index = reordered_mentions.index(m)
            mention_number_previous_mentions[m] = m_index
        mention_number_previous_mentions = {str(k): v for k, v in mention_number_previous_mentions.items()}
        return mention_number_previous_mentions

    def number_of_previous_entities (self): 
        "number of entities in the previous context"
        
        mention_number_previous_entities = {}
        previous_mentions = self.mention_previous_mentions ()
        target_mentions = self.target_mentions
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example['clusters']] 
        for m in target_mentions:  
            number_previous_entities = 0
            for cluster in gold_clusters:
                if len(list(set(previous_mentions[m]).intersection(cluster))) != 0:
                    number_previous_entities += 1  
            mention_number_previous_entities[m] = number_previous_entities
        mention_number_previous_entities = {str(k): v for k, v in mention_number_previous_entities.items()}
        return mention_number_previous_entities
  
  
       
    def entity_prior_reference_frequency (self) :
        "times that the current entity has been referred to in the previous context"
        
        entity_prior_reference_frequency = {}
        previous_mentions = self.mention_previous_mentions ()
        target_mentions = self.target_mentions
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example['clusters']] 
        mention_to_gold = {}
        for gc in gold_clusters:
              for mention in gc:
                  mention_to_gold[mention] = gc      
        for m in target_mentions:
            entity_prior_reference_frequency[m] = len(list(set(previous_mentions[m]).intersection(mention_to_gold[m])))
        entity_prior_reference_frequency = {str(k): v for k, v in entity_prior_reference_frequency.items()}
        return entity_prior_reference_frequency 
 
    def entity_topicality (self) :
        "entity_topicality = entity_prior_reference_frequency / number_of_previous_mentions" 
        
        entity_topicality = {}
        target_mentions = self.target_mentions
        entity_prior_referred_time = self.entity_prior_reference_frequency ()
        number_of_previous_mentions = self.number_of_previous_mentions()
        for m in target_mentions:
            if entity_prior_referred_time[str(m)] == 0 or number_of_previous_mentions[str(m)] == 0:
                entity_topicality[m] = 0
            else:
                entity_topicality[m] = entity_prior_referred_time[str(m)]/number_of_previous_mentions[str(m)]
        entity_topicality = {str(k): v for k, v in entity_topicality.items()}
        return entity_topicality   

    
    def entity_topicality_by_previous_tokens(self):
        "entity_topicality_by_previous_tokens = entity_prior_reference_frequency / number of previous tokens"
        
        entity_topicality_by_tokens = {}
        target_mentions = self.target_mentions
        entity_prior_referred_time = self.entity_prior_reference_frequency ()
        number_of_previous_tokens = self.mention_previous_tokens()
        for m in target_mentions:
            if entity_prior_referred_time[str(m)] == 0 or number_of_previous_tokens[m] == 0:
                entity_topicality_by_tokens[m] = 0
            else:
                entity_topicality_by_tokens[m] = entity_prior_referred_time[str(m)]/number_of_previous_tokens[m]
        return entity_topicality_by_tokens   
    
    

    def antecedent_grammatical_function (self) :
        "the grammatical function of the gold cloest antecedent: subject, non-subject, none"
        
        def grammatical_subject_in_the_tree (tree):
            try:
                for st in tree.subtrees():           
                    for ch in range(len(st)):
                        if st[ch].label() == 'NP' and any(x for x in st[ch+1:] if x.label()=='VP') == True:
                            NP = st[ch]
                            after_NP = st[ch+1]                            
                            return NP, after_NP                
                        elif (st[ch].label() == 'VP' and st[ch+1].label() == 'NP'):
                            NP = st[ch+1]  
                            after_NP = st[ch+2] 
                            return NP,after_NP
            except (AttributeError, IndexError):
                pass
                        
        mention_recent_antecedent = self.mention_last_antecedent()
        mention_sentence_subtoken_map = self.mention_to_sentence_subtoken() 
        list_sentences = self.where_sentences_in_ontonotes()
        mention_strings = self.mention_to_string_mention()
        antecedent_grammatical_function = {}
        mention_antecedent_sentence_distance = self.mention_antecedent_sentence_distance ()
        for m in self.target_mentions:
            ant = mention_recent_antecedent[m] #antecedent
            if ant == 'none':
                antecedent_grammatical_function[m] = 'no_antecedent'
            else:
                ant_sentence_index = mention_sentence_subtoken_map[ant][0]   # sentence where the antecedent is
                ant_sentence = list_sentences[ant_sentence_index]
                ant_token_index = mention_sentence_subtoken_map[ant][1]
                ant_sentence_parse_tree = ant_sentence.parse_tree
                try:
                    after_ant_token = ant_sentence.words[ant_token_index[1]+1]
                    if grammatical_subject_in_the_tree (ant_sentence_parse_tree) == None:
                        antecedent_grammatical_function[m] = 'non_subject'
                    else:
                        subject_in_ant_sentence = grammatical_subject_in_the_tree (ant_sentence_parse_tree)[0]
                        after_subject = grammatical_subject_in_the_tree (ant_sentence_parse_tree)[1]
                        after_subject_token = after_subject.leaves()[0]
                        ant_strings = mention_strings[str(ant)]
                        subject_strings = ' '.join(subject_in_ant_sentence.leaves())
                        if (ant_strings == subject_strings and after_ant_token == after_subject_token) and mention_antecedent_sentence_distance[str(m)] < 2:
                            antecedent_grammatical_function[m] = 'subject'
                        else:
                            antecedent_grammatical_function[m] = 'non_subject'
                except (AttributeError, IndexError):
                    input_doc_key = self.example['doc_key']
                    antecedent_grammatical_function[m] = 'unknown_' + input_doc_key
                    print('grammatical_function Error: doc - ', input_doc_key, "ant_strings:", mention_strings[str(ant)])
        antecedent_grammatical_function = {str(k): v for k, v in antecedent_grammatical_function.items()}
        return antecedent_grammatical_function
    
    
    
    
    

    def mention_grammatical_function (self) :
        "the grammatical function of a mention: subject, non-subject, none"
        
        def grammatical_subject_in_the_tree (tree):
            try:
                for st in tree.subtrees():           
                    for ch in range(len(st)):
                        if st[ch].label() == 'NP' and any(x for x in st[ch+1:] if x.label()=='VP') == True:
                            NP = st[ch]
                            after_NP = st[ch+1]                            
                            return NP, after_NP                
                        elif (st[ch].label() == 'VP' and st[ch+1].label() == 'NP'):
                            NP = st[ch+1]  
                            after_NP = st[ch+2] 
                            return NP,after_NP
            except (AttributeError, IndexError):
                pass
        
        mention_sentence_subtoken_map = self.mention_to_sentence_subtoken()
        # masked setup: {new mention indices: original sentence subtoken map}
        list_sentences = self.where_sentences_in_ontonotes()
        mention_to_string = self.mention_to_string_mention()
        
        current_mention_grammatical_function = {}
        
        for m in self.target_mentions:

            mention_sentence_index = mention_sentence_subtoken_map[m][0]   # sentence where the antecedent is
            mention_sentence = list_sentences[mention_sentence_index]
            mention_token_index = mention_sentence_subtoken_map[m][1]
            mention_sentence_parse_tree = mention_sentence.parse_tree
            try:
                after_mention_token = mention_sentence.words[mention_token_index[1]+1]
                if grammatical_subject_in_the_tree (mention_sentence_parse_tree) == None:
                    current_mention_grammatical_function[m] = 'non_subject'
                else:
                    subject_in_mention_sentence = grammatical_subject_in_the_tree (mention_sentence_parse_tree)[0]
                    after_subject = grammatical_subject_in_the_tree (mention_sentence_parse_tree)[1]
                    after_subject_token = after_subject.leaves()[0]
                    mention_strings = mention_to_string[str(m)]
                    subject_strings = ' '.join(subject_in_mention_sentence.leaves())
                    if (mention_strings == subject_strings and after_mention_token == after_subject_token):
                        current_mention_grammatical_function[m] = 'subject'
                    else:
                        current_mention_grammatical_function[m] = 'non_subject'
            except (AttributeError, IndexError):
                input_doc_key = self.example['doc_key']
                current_mention_grammatical_function[m] = 'non_subject'
                print('grammatical_function Error: doc - ', input_doc_key, "mention_strings:", mention_to_string[str(m)])
        return current_mention_grammatical_function    
    
       
        
        
    def mention_antecedent_sentence_distance (self):
        "number of sentences between a mention and its gold closest antecedent"
        # 0: mention and its gold antecedent are in the same sentence
        # 1: the gold antecedent appears in the sentence immediately preceding the current mention
        
        mention_recent_antecedent = self.mention_last_antecedent()
        mention_sentence_subtoken_map = self.mention_to_sentence_subtoken() 
        mention_antecedent_sentence_distance = {}
        for m in self.target_mentions:
            ant = mention_recent_antecedent[m] #antecedent
            if ant == 'none':
                mention_antecedent_sentence_distance[m] = 'no_antecedent'
            else:
                mention_sentence_index = mention_sentence_subtoken_map[m][0] # sentence where the current mention is
                ant_sentence_index = mention_sentence_subtoken_map[ant][0]   # sentence where the antecedent is
                mention_antecedent_sentence_distance[m] = mention_sentence_index - ant_sentence_index
        mention_antecedent_sentence_distance = {str(k): v for k, v in mention_antecedent_sentence_distance.items()}
        return mention_antecedent_sentence_distance
    
    
    
    def mention_token_length (self): 
        mention_to_length = {}
        mention_words_pos = self.mention_to_words_pos () 
        for m in mention_words_pos:
            mention_tokens = mention_words_pos[m][0]
            mention_to_length[m] = len(mention_tokens)
        return mention_to_length


    def softmax(self, x):
        # scipy version (v1.0.0) too low to import softmax
        # the formula is defined here (x is a one-dimensional numpy array)        
        result = np.exp(x)/sum(np.exp(x))
        return result
    
    def surprisal (self, probability):
        result = - np.log2(probability)
        return result
    
    
    def correct_wrong_prediction(self, mention_to_predicted_antecedent, mention_to_gold):
        "prediction from the model is correct or wrong"
        
        antecedent_prediction_quality = {}
        first_mentions = self.list_first_mentions()   
        for m in mention_to_predicted_antecedent:
            if mention_to_predicted_antecedent[m] == 'none':
                if m in first_mentions:
                    antecedent_prediction_quality[m] = "correct"
                else:
                    antecedent_prediction_quality[m] = "wrong"                
            elif mention_to_predicted_antecedent[m] in mention_to_gold[m]:
                antecedent_prediction_quality[m] = "correct"
            
            else:
                antecedent_prediction_quality[m] = "wrong"
        return antecedent_prediction_quality 
    
    def gold_antecedent_masked_or_not (self, mention_to_gold_antecedent, list_masked_mentions):
        "to check: the closest gold antecedent is masked or not when the model is doing the prediction"
        
        gold_antecedent_masked_or_not = {}
        first_mentions = self.list_first_mentions()
        for m in mention_to_gold_antecedent:
            if m in first_mentions:
                gold_antecedent_masked_or_not[m] = 'no_antecedent'
            elif mention_to_gold_antecedent[m] in list_masked_mentions:
                gold_antecedent_masked_or_not[m] = 'masked'
            else:
                gold_antecedent_masked_or_not[m] = 'unmasked'
            
        return gold_antecedent_masked_or_not
    
    
    def probability_distribution(self, gold_mentions, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores):
        # the first antecedent: 'none'
        # the first score prior softmax: 0 
        
        predictions = {}
        predictions["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
        predictions["top_span_starts"] = top_span_starts.tolist()
        predictions["top_span_ends"] = top_span_ends.tolist()
        predictions["top_antecedents"] = list(zip(predictions["top_spans"],[list(zip(["none"] + [predictions["top_spans"][i] for i in row1], row2.astype(float).tolist())) for row1, row2 in zip(top_antecedents, top_antecedent_scores)]))
        def extract_antecedent_scores_as_dicts(mention, antecedent_scores):
            result = []
            for antecedent, score in antecedent_scores:
                if score != float('-inf'):
                    result.append(
                        {'mention': mention,
                         'antecedent': antecedent,
                         'score': score,
                         }
                    )
            return result    
        result = []
        for mention, antecedent_scores in predictions["top_antecedents"]:
            if mention in gold_mentions: # only consider mentions in gold_mentions 
                result += extract_antecedent_scores_as_dicts(mention, antecedent_scores)
        new_result = sorted(result, key = operator.itemgetter("mention"))
        grouped_list = []
        for i,g in itertools.groupby(new_result, key=operator.itemgetter("mention")):
            grouped_list.append(list(g)) # a list of list of dictionaries
        pro_dic = [] # a list of dictionaries with "mention", ""
        for l in grouped_list:
            mention_dic = {}
            mention_dic["mention"] = l[0]["mention"]
            mention_dic["antecedents"] = []
            mention_dic["scores"] = []
            for dic in l:
                mention_dic["antecedents"].append(dic["antecedent"])
                mention_dic["scores"].append(dic["score"])
            
            mention_dic["scores"] = self.softmax(mention_dic["scores"])
            pro_dic.append(mention_dic)
        
        return pro_dic
    
 
    def probability_gold_antecedent(self, dict_gold_antecedent, probability_distribution):
        "probability the model assigns to the gold closest antecedent"
        # the gold_antecedent of the first mention is 'none'
        
        dict_probability_gold_antecedent = {}
        for d in probability_distribution:            
            mention = d['mention']
            gold_antecedent = dict_gold_antecedent[mention]
            antecedents = d['antecedents'] # the first antecedent in the list is 'none'
            if gold_antecedent in antecedents:
                pro_dist = d['scores']
                index = antecedents.index(gold_antecedent)
                gold_antecedent_probability = pro_dist[index]
                dict_probability_gold_antecedent[mention] = gold_antecedent_probability                                          
            else:
                dict_probability_gold_antecedent[mention] = 'gold_antecedent_not_detected'
                print('gold_antecedent_not_detected')
                print("mention:", mention)
                print("antecedents:", antecedents)
                print("gold_antecedent:", gold_antecedent)
        return dict_probability_gold_antecedent

    def list_first_mentions(self) :
        "list of all mentions (indices) that are the first mention of an entity"
        
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example["clusters"]]
        first_mentions = []
        for gc in gold_clusters:
            first_mention = sorted(gc, key= lambda x: x[0])[0] 
            first_mentions.append(first_mention)  
        return first_mentions
    
    def probability_gold_entity(self, mention_to_gold, probability_distribution):
        "the sum of probabilities the model assigns to gold antecedents"
        
        dict_probability_gold_entity = {}
        detected_gold_antecedents = {}  
        first_mentions = self.list_first_mentions()
        for d in probability_distribution:
            mention = d['mention']
            gold_antecedents = mention_to_gold[mention] # gold cluster
            antecedents_scores = d["scores"]
            predicted_antecedents = d["antecedents"]
            probability = 0
            detected_antecedents = []
            if mention in first_mentions:
            # probability_gold_entity = probability_gold_antecedent = probability of "none"
                  probability = self.probability_gold_antecedent(self.mention_last_antecedent(), probability_distribution)[mention]
                  detected_antecedents.append('none')
                  detected_gold_antecedents[mention] = 1

            else:
                for i,predicted_a in enumerate(predicted_antecedents):             
                    if predicted_a in gold_antecedents:
                        probability += antecedents_scores[i]
                        detected_antecedents.append(predicted_a)
                detected_gold_antecedents[mention] = len(detected_antecedents)

            if len(detected_antecedents) == 0:
                dict_probability_gold_entity[mention] = "gold_antecedents_not_detected"
            else:    
                dict_probability_gold_entity[mention] = probability
        return dict_probability_gold_entity, detected_gold_antecedents
            
                                   
    
    def probability_predicted_antecedent(self, mention_to_predicted, probability_distribution):
        "probability the model assigns to its predicted antecedent"
        
        dict_probability_predicted_antecedent = {}
        for d in probability_distribution:
            mention = d['mention']
            predicted_antecedent = mention_to_predicted[mention] # mention_to_predicted contains 'none' as guess for no_antecedent
            if predicted_antecedent == 'none':
                dict_probability_predicted_antecedent[mention] = d['scores'][0] 
            else:
                antecedents = d['antecedents']
                pro_dist = d['scores']
                index = antecedents.index(predicted_antecedent)
                predicted_antecedent_probability = pro_dist[index]
                dict_probability_predicted_antecedent[mention] = predicted_antecedent_probability
                            
        return dict_probability_predicted_antecedent
    
    def probability_predicted_entity(self, mention_to_predicted, probability_distribution):
        "the sum of probabilities the model assigns to all mentions referring to the predicted entity"
        
        dict_probability_predicted_entity = {}
        dict_detected_predicted_antecedents = {}
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example["clusters"]]
        mention_to_gold = {}
        for gc in gold_clusters:
            for m in gc: 
                mention_to_gold[m] = gc                                                       
        for dic in probability_distribution:
            mention = dic['mention']
            predicted_antecedent = mention_to_predicted[mention] # 'none' as the guess for no_antecedent
            
            if predicted_antecedent == 'none':
                dict_probability_predicted_entity[mention] = self.probability_predicted_antecedent(mention_to_predicted, probability_distribution)[mention]          
            elif predicted_antecedent in mention_to_gold:
                predicted_antecedent_cluster = mention_to_gold[predicted_antecedent]
                detected_antecedents = dic['antecedents']
                antecedents_scores = dic['scores']
                probability = 0
                detected_predicted_antecedents = []
                for i, detected_a in enumerate(detected_antecedents):
                    if detected_a in predicted_antecedent_cluster:
                        probability += antecedents_scores[i]
                        detected_predicted_antecedents.append(detected_a)
                dict_probability_predicted_entity[mention] = probability
                dict_detected_predicted_antecedents[mention] = detected_predicted_antecedents
            else: # predicted_antecedent is not in gold_mentions
                dict_probability_predicted_entity[mention] = 'predicted_antecedent_not_in_gold_mention'
                dict_detected_predicted_antecedents[mention] = 'predicted_antecedent_not_in_gold_mention'
        return dict_probability_predicted_entity, dict_detected_predicted_antecedents
    
    
    
    def entropy_distribution(self, probability_distribution):
        "entropy of probability distribution over all previous mentions"
        
        dict_entropy = {}
        num_of_mentions_probability_distributed_over = {}
        for dic in probability_distribution:
            mention = dic["mention"]
            antecedents_scores = dic["scores"]
            dict_entropy[mention] = entropy(antecedents_scores)
            num_of_mentions_probability_distributed_over[mention] = len(antecedents_scores) # including one antecedent named "none"
        return dict_entropy, num_of_mentions_probability_distributed_over
    
    
    
    def entropy_distribution_over_entities_excluding_gold_antecedent (self, gold_mentions, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores): 
        dict_entropy_over_other_entities = {}
        mention_to_gold_antecedent = self.mention_last_antecedent()
        first_mentions = self.list_first_mentions()
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example["clusters"]]
        # the first antecedent: 'none'
        # the first score prior softmax: 0 
        predictions = {}
        predictions["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
        predictions["top_span_starts"] = top_span_starts.tolist()
        predictions["top_span_ends"] = top_span_ends.tolist()
        predictions["top_antecedents"] = list(zip(predictions["top_spans"],[list(zip(["none"] + [predictions["top_spans"][i] for i in row1], row2.astype(float).tolist())) for row1, row2 in zip(top_antecedents, top_antecedent_scores)]))
        def extract_antecedent_scores_as_dicts(mention, antecedent_scores):
            result = []
            for antecedent, score in antecedent_scores:
                if score != float('-inf'):
                    result.append(
                        {'mention': mention,
                         'antecedent': antecedent,
                         'score': score,
                         }
                    )
            return result    
        result = []
        for mention, antecedent_scores in predictions["top_antecedents"]:
            if mention in gold_mentions: # only consider mentions in gold_mentions 
                result += extract_antecedent_scores_as_dicts(mention, antecedent_scores)
        new_result = sorted(result, key = operator.itemgetter("mention"))
        grouped_list = []
        for i,g in itertools.groupby(new_result, key=operator.itemgetter("mention")):
            grouped_list.append(list(g)) # a list of list of dictionaries
        pro_dic = [] # a list of dictionaries with "mention", ""
        for l in grouped_list:
            mention_dic = {}
            mention_dic["mention"] = l[0]["mention"]
            mention_dic["antecedents"] = []
            mention_dic["scores"] = []
            for dic in l:
                mention_dic["antecedents"].append(dic["antecedent"])
                mention_dic["scores"].append(dic["score"])
            
            # mention_dic["scores"] = self.softmax(mention_dic["scores"])
            pro_dic.append(mention_dic)
            
        
        for dic in pro_dic: 
            mention = dic['mention']
            detected_antecedents = dic['antecedents']
            antecedents_scores = dic['scores']
            gold_antecedent = mention_to_gold_antecedent[mention]
            
            if mention in first_mentions:
                detected_antecedents.remove(detected_antecedents[0])
                antecedents_scores.remove(antecedents_scores[0])
            else:
                gold_entity_cluster = [gc for gc in gold_clusters if mention in gc][0]
                for i, detected_a in enumerate(detected_antecedents):                
                    if detected_a in gold_entity_cluster:
                        detected_antecedents.remove(detected_antecedents[i])
                        antecedents_scores.remove(antecedents_scores[i])                                                                                           
            antecedents_scores = self.softmax(antecedents_scores)            
            if len(detected_antecedents) == 0:
                dict_entropy_over_other_entities[mention] = 'no_other_entities'
            
            else:
                new_probability_distribution_over_other_entities = []
                if mention not in first_mentions:
                    # probability of 'none'
                    new_probability_distribution_over_other_entities.append(antecedents_scores[0]) 
                for gd in gold_clusters:
                    if mention not in gd: # exclude the gold entity cluster
                        probability_for_this_cluster = 0
                        for i, detected_a in enumerate(detected_antecedents):                
                            if detected_a in gd:
                                probability_for_this_cluster += antecedents_scores[i] 
                                
                        new_probability_distribution_over_other_entities.append(probability_for_this_cluster)            
                dict_entropy_over_other_entities[mention] = entropy(new_probability_distribution_over_other_entities)            
                                    
        return dict_entropy_over_other_entities        
        
         
    def entropy_distribution_over_entities(self, probability_distribution):
        "entropy of probability distribution over all previous entities"
        # only when given gold mentions
        # "none" is included as another independent entity           
        dict_entropy_over_entities = {}
        first_mentions = self.list_first_mentions()
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in self.example["clusters"]]
        for dic in probability_distribution:
            new_probability_distribution_over_entities = []
            mention = dic['mention']
            antecedents_scores = dic['scores']
            detected_antecedents = dic['antecedents']
            new_probability_distribution_over_entities.append(antecedents_scores[0]) # probability of 'none'
            for gd in gold_clusters:
                probability_for_this_cluster = 0
                for i, detected_a in enumerate(detected_antecedents):                
                    if detected_a in gd:
                        probability_for_this_cluster += antecedents_scores[i]                    
                new_probability_distribution_over_entities.append(probability_for_this_cluster)            
            dict_entropy_over_entities[mention] = entropy(new_probability_distribution_over_entities)
        return dict_entropy_over_entities
                
                   
    
    def surprisal_gold_antecedent(self, probability_gold_antecedent):
        dict_surprisal_gold_antecedent = {}
        for m in probability_gold_antecedent:
            if type(probability_gold_antecedent[m]) == str:
                dict_surprisal_gold_antecedent[m] = "gold_antecedent_not_detected"
            else:            
                dict_surprisal_gold_antecedent[m] = self.surprisal(probability_gold_antecedent[m])
        return dict_surprisal_gold_antecedent
            
        
    def surprisal_gold_entity (self, probability_gold_entity):
        dict_surprisal_gold_entity = {}
        for m in probability_gold_entity:
            if probability_gold_entity[m] == 'gold_antecedents_not_detected' :
                dict_surprisal_gold_entity[m] = "gold_antecedents_not_detected"           
            else:
                dict_surprisal_gold_entity[m] = self.surprisal(probability_gold_entity[m])            
        return dict_surprisal_gold_entity
    

        
        