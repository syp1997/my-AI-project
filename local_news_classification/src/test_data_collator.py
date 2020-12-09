import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import time 
import csv
import math
import logging

logger = logging.getLogger(__name__)


class TestDataCollator():
    """ Load test data """
    
    def __init__(self, data_list=None, data_file=None):
        self.data_list = data_list
        self.data_file = data_file

    def split_data(self):
        if self.data_list is None:
            f = open(self.data_file, 'r')
            data_reader = csv.reader(f, delimiter='\t')
        else:
            data_reader = self.data_list
        docid_list = []
        text_list = []
        entity_list = []
        keyword_list = []
        domain_list = []
        cat_1_list = []
        cat_2_list = []
        cat_3_list = []
        for data in data_reader:
            docid_list.append(data[0])
            text_list.append(data[1])
            entity_list.append(data[2].split('|'))
            keyword_list.append(data[3].split('|'))
            domain_list.append(data[4])
            cat_1_list.append(data[5])
            cat_2_list.append(data[6].split('|'))
            cat_3_list.append(data[7].split('|'))
        return text_list, entity_list, keyword_list, domain_list, (cat_1_list,cat_2_list,cat_3_list)
    
    def encode_text(self,tokenizer):
        text_list = self.split_data()[0]
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
    
    def encode_entity(self, wiki2vec, idf_dict, unk_idf, en_pad_size, en_embd_dim, entity_score_dict, keyword_score_dict,
                      keyword_entropy_dict, domain_score_dict, keyword_entropy_mean, cat_1_dict, cat_2_dict, cat_3_dict):
        _, entity_list, keyword_list, domain_list, cat_list = self.split_data()
        entity_vectors = []
        entity_length = []
        entity_score = []
        for entities in entity_list:
            # conpute entity vector
            nopad_vectors=[]
            for entity in entities[:en_pad_size]:
                nopad_vectors.append(
                    torch.tensor(self.compute_entity_vector(entity, wiki2vec, idf_dict, unk_idf, en_embd_dim)).float()
                )
            nopad_vectors = torch.stack(nopad_vectors)
            # pad entity vector
            vectors = torch.cat([nopad_vectors,torch.zeros(en_pad_size-nopad_vectors.shape[0], en_embd_dim)])
            entity_vectors.append(vectors)
            
            # record entity length
            entity_length.append(len(entities))
            
            # compute entity score
            score_list = []
            count_list = []
            for en in entities:
                if en in entity_score_dict:
                    en_score = float(entity_score_dict[en][0])
                    en_count = int(entity_score_dict[en][1])
                    score_list.append(en_score)
                    count_list.append(en_count)
            mean_count = np.mean(count_list)
            score = 1
            for i in range(len(score_list)):
                score_list[i]=score_list[i] ** np.sqrt(count_list[i]/mean_count)
                score *= score_list[i]
            score = math.log(score+1e-12,10)
            if len(score_list) == 0:
                max_score = min_score = 0
            else:
                max_score = math.log(max(score_list)**len(score_list)+1e-12,10)
                min_score = math.log(min(score_list)**len(score_list)+1e-12,10)
            entity_score.append([score,max_score,min_score])
        entity_vectors = torch.stack(entity_vectors)
        entity_length = torch.tensor(entity_length)
        entity_score = torch.tensor(entity_score)
            
        keyword_score = []
        keyword_entropy = []
        for keywords in keyword_list:
            # compute keyword score
            score_list = []
            count_list = []
            for kws in keywords:
                if kws in keyword_score_dict:
                    kws_score = float(keyword_score_dict[kws][0])
                    kws_count = int(keyword_score_dict[kws][1])
                    score_list.append(kws_score)
                    count_list.append(kws_count)
            mean_count = np.mean(count_list)
            score = 1
            for i in range(len(score_list)):
                score_list[i]=score_list[i] ** np.sqrt(count_list[i]/mean_count)
                score *= score_list[i]
            score = math.log(score+1e-12,10)
            if len(score_list) == 0:
                max_score = min_score = 0
            else:
                max_score = math.log(max(score_list)**len(score_list)+1e-12,10)
                min_score = math.log(min(score_list)**len(score_list)+1e-12,10)
            keyword_score.append([score,max_score,min_score])
            
            # compute keyword entropy
            entropy = 0
            num_words = 0
            for word in keywords:
                if word in keyword_entropy_dict:
                    num_words += 1
                    word_entropy = float(keyword_entropy_dict[word])
                    entropy += word_entropy
            if num_words == 0:
                keyword_entropy.append(math.exp(keyword_entropy_mean))
            else:
                keyword_entropy.append(math.exp(entropy/len(keywords)))
        keyword_score = torch.tensor(keyword_score)
        keyword_entropy = torch.tensor(keyword_entropy)
            
        # compute domain score
        domain_score = []
        for domain in domain_list:
            score = 1.0
            if domain in domain_score_dict:
                score = float(domain_score_dict[domain])
            domain_score.append(score)
        domain_score = torch.tensor(domain_score)
        
        # compute category sccore
        cat_1_list,cat_2_list,cat_3_list = cat_list
        all_cat_score = []
        for cat_1,cat_2,cat_3 in zip(cat_1_list,cat_2_list,cat_3_list):
            cat_1_score = 1
            cat_2_score = 1
            cat_3_score = 1
            if cat_1 in cat_1_dict:
                cat_1_score = float(cat_1_dict[cat_1])
                cat_1_score = math.log(cat_1_score+1e-12,10)
            for cat in cat_2:
                if cat in cat_2_dict:
                    score = float(cat_2_dict[cat])
                    cat_2_score *= score
            cat_2_score = math.log(cat_2_score+1e-12,10)
            for cat in cat_3:
                if cat in cat_3_dict:
                    score = float(cat_3_dict[cat])
                    cat_3_score *= score
            cat_3_score = math.log(cat_3_score+1e-12,10)    
            cat_score=[cat_1_score,cat_2_score,cat_3_score]
            all_cat_score.append(cat_score)
        all_cat_score = torch.tensor(all_cat_score)
        
        return entity_vectors, entity_length, entity_score, keyword_score, keyword_entropy, domain_score, all_cat_score
    
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
    
    def en_vector_norm(self, vector):
        # normlize entity vector
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-9)
    
    def load_data(self, batch_size, tokenizer, wiki2vec, idf_dict, unk_idf, en_pad_size, en_embd_dim, 
                  entity_score_dict, keyword_score_dict, keyword_entropy_dict, domain_score_dict, 
                  keyword_entropy_mean, cat_1_dict, cat_2_dict, cat_3_dict):
        # build dataset and load dataloader
        last_time = time.time()
        input_ids = self.encode_text(tokenizer)
        logger.info('Encode text: Took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        features = self.encode_entity(
            wiki2vec, idf_dict, unk_idf, 
            en_pad_size, en_embd_dim, 
            entity_score_dict, keyword_score_dict, 
            keyword_entropy_dict, domain_score_dict, 
            keyword_entropy_mean, cat_1_dict, 
            cat_2_dict, cat_3_dict
        )
        entity_vectors, entity_length, entity_score, keyword_score, keyword_entropy, domain_score, cat_score = features
        logger.info('Encode other features: Took {} seconds'.format(time.time() - last_time))
        # Split data into train and validation
        dataset = TensorDataset(input_ids, entity_vectors, entity_length, entity_score, 
                                keyword_score, keyword_entropy, domain_score, cat_score)
        
        # Create train and validation dataloaders
        test_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        
        return test_dataloader