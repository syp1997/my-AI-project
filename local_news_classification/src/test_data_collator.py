import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import time 
import csv
import math
import logging

logger = logging.getLogger(__name__)


class TestDataCollator():
    
    def __init__(self, test_data_file):
        self.test_data_file = test_data_file

    def split_data(self):
        docid_list = []
        text_list = []
        entity_list = []
        with open(self.test_data_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                docid_list.append(line[0])
                text_list.append(line[1])
                entity_list.append(list(set(line[2].split('|'))))
        return text_list, entity_list

    # Function to get token ids for a list of texts 
    def encode_text(self,tokenizer):
        text_list, _ = self.split_data()
        all_input_ids = []    
        num = 0
        for text in text_list:
            input_ids = tokenizer.encode(
                            text,                      
                            add_special_tokens = True,             
                            truncation=True,
                            padding = 'max_length',     
                            return_tensors = 'pt'       
                       )
            all_input_ids.append(input_ids)    
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids
    
    # Function to get token ids for a list of entities
    def encode_entity(self, entity_to_index, index_to_entity, wiki2vec, idf_dict, unk_idf, 
                      en_pad_size, en_embd_dim, entity_score_dict):
        _, entity_list = self.split_data()
        entity_vectors = []
        entity_length = []
        entity_score = []
        for entities in entity_list:
            # conpute entity vector
            nopad_vectors=[]
            for entity in entities[:en_pad_size]:
                nopad_vectors.append(torch.tensor(self.compute_entity_vector(entity, wiki2vec, idf_dict, unk_idf, en_embd_dim)).float())
            nopad_vectors = torch.stack(nopad_vectors)
            # pad entity vector
            vectors = torch.cat([nopad_vectors,torch.zeros(en_pad_size-nopad_vectors.shape[0], en_embd_dim)])
            entity_vectors.append(vectors)
            
            # record entity length
            entity_length.append(len(entities))
            
            # compute entity score
            score = 1
            for en in entities:
                if en in entity_score_dict:
                    en_score = float(entity_score_dict[en])
                    score *= en_score
            score = math.log(score+1e-12,10)
            entity_score.append(score)
            
        entity_vectors = torch.stack(entity_vectors)
        entity_length = torch.tensor(entity_length)
        entity_score = torch.tensor(entity_score)
        
        return entity_vectors, entity_length, entity_score
    
    def compute_entity_vector(self, entity, wiki2vec, idf_dict, unk_idf, en_embd_dim):
        entity_item = wiki2vec.get_entity(entity)
        if entity_item != None:
            return self.en_vector_norm(wiki2vec.get_vector(entity_item))
        else:
            words = entity.lower().split()
            word_vectors = []
            weights = []
            for w in words:
                try:
                    vector = wiki2vec.get_word_vector(w.lower())
                except KeyError:
                    continue
                word_vectors.append(vector)
                idf = idf_dict.get(w, unk_idf)
                weights.append(idf)
            if len(word_vectors) == 0:
                return np.zeros(en_embd_dim)
            else:
                word_vectors = np.array(word_vectors)
                weights = np.expand_dims(np.array(weights), axis=1)
                return self.en_vector_norm(np.sum(word_vectors * weights, axis=0))
    
    # normlize entity vector
    def en_vector_norm(self, vector):
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-9)
    
    
    # build dataset and dataloader
    def load_data(self, batch_size, tokenizer, entity_to_index, index_to_entity, wiki2vec, idf_dict, unk_idf, 
                  en_pad_size, en_embd_dim, entity_score_dict):
        last_time = time.time()
        input_ids = self.encode_text(tokenizer)
        logger.info('Encode text: Took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        entity_vectors, entity_length, entity_score = self.encode_entity(entity_to_index, index_to_entity, wiki2vec, idf_dict, unk_idf, en_pad_size, en_embd_dim, entity_score_dict)
        logger.info('Encode entity: Took {} seconds'.format(time.time() - last_time))
        # Split data into train and validation
        dataset = TensorDataset(input_ids, entity_vectors, entity_length, entity_score)
        
        # Create train and validation dataloaders
        test_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        
        return test_dataloader