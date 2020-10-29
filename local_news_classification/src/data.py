import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import csv
import collections
from collections import Counter
import math


class DataProcess():
    
    def __init__(self, data_root, text_id_root, labels_root, entity_id_root, entity_length_root, entity_score_root):
        self.data_root = data_root
        self.text_id_root = text_id_root
        self.labels_root = labels_root
        self.entity_id_root = entity_id_root
        self.entity_length_root = entity_length_root
        self.entity_score_root = entity_score_root
    
    def prepare_data(self):
        docid_list = []
        text_list = []
        entity_list = []
        label_list = []
        with open(self.data_root, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                docid_list.append(line[0])
                text_list.append(line[1])
                entity_list.append(list(set(line[2].split('|'))))
                label_list.append(int(line[3]))
        return text_list, entity_list, label_list

    # Function to get token ids for a list of texts 
    def encode_text(self, tokenizer):
        text_list, _, label_list = self.prepare_data()
        all_input_ids = []    
        num = 0
        for text in text_list:
            num += 1
            if num % 10000 == 0:
                print("Processing Data: ", num)
            input_ids = tokenizer.encode(
                            text,                      
                            add_special_tokens = True,             
                            truncation=True,
                            padding = 'max_length',     
                            return_tensors = 'pt'       
                       )
            all_input_ids.append(input_ids)    
        all_input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.tensor(label_list, dtype=torch.float)
        # Save tensor
        torch.save(all_input_ids, self.text_id_root)
        torch.save(labels,self.labels_root)
        print("Saved success!")
        return all_input_ids, labels
    
    # Function to build entity vocab
    def encode_entity(self):
        _, entity_list, _ = self.prepare_data()
        # get all entity
        entity_list_all = [en for entity in entity_list for en in entity]
        print("All Entity number: ", len(entity_list_all))
        # build entity vocab
        entity_vocab = collections.OrderedDict(Counter(entity_list_all))
        entity_list_uniq = [entity for entity in entity_vocab.keys()]
        entity_to_index = {entity : i+2 for i, entity in enumerate(entity_list_uniq)}
        entity_to_index['<unk>'] = 0
        entity_to_index['<pad>'] = 1
        entity_to_index = collections.OrderedDict(sorted(entity_to_index.items(), key=lambda entity_to_index: entity_to_index[1]))
        index_to_entity = [entity for i, entity in enumerate(entity_to_index)]
        print("Entity vocab size: ", len(entity_to_index))
        return entity_to_index, index_to_entity
    
    # Function to build entity vocab with pretrained vector
    def build_entity_vector(self, entity_to_index, index_to_entity, wiki2vec, idf_dict, unk_idf, en_embd_dim, entity_vector_root):
        # build entity vector
        idx_to_vector=[]
        for entity in entity_to_index.keys():
            entity_item = wiki2vec.get_entity(entity)
            if entity_item != None:
                idx_to_vector.append(torch.tensor(self.en_vector_norm(wiki2vec.get_vector(entity_item))).float())
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
                    idx_to_vector.append(torch.zeros(en_embd_dim))
                else:
                    word_vectors = np.array(word_vectors)
                    weights = np.expand_dims(np.array(weights), axis=1)
                    idx_to_vector.append(torch.tensor(self.en_vector_norm(np.sum(word_vectors * weights, axis=0))).float())
        entity_vector = torch.stack(idx_to_vector)
        torch.save(entity_vector, entity_vector_root)
        print("Saved success!")
        return entity_vector
    
    # Function to get token ids for a list of entities
    def build_entity_id(self, entity_to_index, index_to_entity, en_pad_size):
        # build entity index
        _, entity_list, _ = self.prepare_data()
        all_entity_ids = []
        all_entity_length = []
        for entities in entity_list:
            entity_ids = [entity_to_index.get(entity, entity_to_index["<unk>"]) for entity in entities][:en_pad_size]
            for i in range(en_pad_size - len(entity_ids)):
                entity_ids.append(entity_to_index["<pad>"])
            all_entity_ids.append(entity_ids)
            # record entity length
            all_entity_length.append(len(entities))
        all_entity_ids = torch.tensor(all_entity_ids)
        all_entity_length = torch.tensor(all_entity_length)
        torch.save(all_entity_ids, self.entity_id_root)
        torch.save(all_entity_length, self.entity_length_root)
        print("Saved success!")
        return all_entity_ids, all_entity_length
    
    def build_entity_score(self, entity_score_dict):
        _, entity_list, _ = self.prepare_data()
        all_entity_score = []
        for entities in entity_list:
            entity_score = []
            score = 1
            for en in entities:
                if en in entity_score_dict:
                    en_score = float(entity_score_dict[en])
                    score *= en_score
            score = math.log(score,10)
            entity_score.append(score)
            if score >= 0:
                entity_score.append(score**2)
                entity_score.append(score**0.5)
            else:
                entity_score.append(-score**2)
                entity_score.append(-(abs(score)**0.5))
            all_entity_score.append(entity_score)
        all_entity_score = torch.tensor(all_entity_score)
        all_entity_score, entity_score_mean, entity_score_std = self.en_score_norm(all_entity_score)
        torch.save(all_entity_score, self.entity_score_root)
        print("Entity score mean: ", entity_score_mean)
        print("Entity score std: ", entity_score_std)
        return all_entity_score, entity_score_mean, entity_score_std
        
    # normlize entity vector
    def en_vector_norm(self, vector):
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-9)
    
    def en_score_norm(self,x):
        mean = x.mean(dim=0,keepdim=True)
        std = x.std(dim=0, unbiased=False,keepdim=True)
        x_norm = (x - mean)/std
        return x_norm, mean, std
    
    # build dataset and dataloader
    def load_data(self, ratio, batch_size):
        all_input_ids = torch.load(self.text_id_root)
        all_entity_ids = torch.load(self.entity_id_root)
        all_entity_length = torch.load(self.entity_length_root)
        all_entity_score = torch.load(self.entity_score_root)
        labels = torch.load(self.labels_root)
        # Split data into train and validation
        dataset = TensorDataset(all_input_ids, all_entity_ids, all_entity_length, all_entity_score, labels)
        train_size = int(ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # Create train and validation dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
        
        print("Num of train_dataloader: ", len(train_dataloader))
        print("Num of valid_dataloader: ", len(valid_dataloader))

        return train_dataloader, valid_dataloader
    
    
    # load idf file
    def load_idf(self, idf_file):
        ret = {}
        with open(idf_file) as f:
            for line in f:
                phrase, count, idf = line.split('\t')
                idf = float(idf)
                ret[phrase] = idf
        return ret, ret['<UNK>']

    def load_entity_score_dict(self, entity_frep_file, min_count=5):
        entity_score_dict = {}
        with open(entity_frep_file) as f:
            for line in f:
                entity, c1, c2, freq = line.split('\t')
                c1 = int(c1)
                c2 = int(c2)
                if c1 == 0 or c2 == 0:
                    c1 += 1
                    c2 += 1
                if c1 + c2 > min_count:
                    entity_score_dict[entity] = freq
        print("Entity Score vocab size: ", len(entity_score_dict))
        return entity_score_dict
    
    def load_entity_vector(self, entity_vector_root):
        entity_vector = torch.load(entity_vector_root)
        print("Entity vector shape: ", entity_vector.shape)
        return torch.load(entity_vector_root)