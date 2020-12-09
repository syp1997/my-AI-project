import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import csv
import collections
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)


class DataProcess():
    """ Process data especially training data. """
    
    def __init__(self, train_data_file=None):
        # input train data file when need training model
        self.train_data_file = train_data_file
    
    def prepare_data(self):
        # split data: docid, text, entity, keywords, domain, (cat_1, cat_2, cat_3), label
        docid_list = []
        text_list = []
        entity_list = []
        keyword_list = []
        domain_list = []
        cat_1_list = []
        cat_2_list = []
        cat_3_list = []
        label_list = []
        with open(self.train_data_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                docid_list.append(line[0])
                text_list.append(line[1])
                entity_list.append(line[2].split('|'))
                keyword_list.append(line[3].split('|'))
                domain_list.append(line[4])
                cat_1_list.append(line[5])
                cat_2_list.append(line[6].split('|'))
                cat_3_list.append(line[7].split('|'))
                label_list.append(int(line[8]))
        return docid_list, text_list, entity_list, keyword_list, domain_list, (cat_1_list,cat_2_list,cat_3_list), label_list

    def encode_text(self, tokenizer, text_id_root, labels_root):
        #get token ids for a bunch of texts
        _, text_list, _, _, _, _, label_list = self.prepare_data()
        all_input_ids = []    
        num = 0
        for text in text_list:
            num += 1
            if num % 10000 == 0:
                logger.info("Processing Data: {}w".format(num/10000))
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
        torch.save(all_input_ids, text_id_root)
        torch.save(labels,labels_root)
        logger.info("Saved success to {} and {}".format(text_id_root,labels_root))
        return all_input_ids, labels
    
    def build_entity_vocab(self):
        entity_list = self.prepare_data()[2]
        # get all entity
        entity_list_all = [en for entity in entity_list for en in entity]
        logger.info("All Entity number: {}".format(len(entity_list_all)))
        # build entity vocab
        entity_vocab = collections.OrderedDict(Counter(entity_list_all))
        entity_list_uniq = [entity for entity in entity_vocab.keys()]
        entity_to_index = {entity : i+2 for i, entity in enumerate(entity_list_uniq)}
        entity_to_index['<unk>'] = 0
        entity_to_index['<pad>'] = 1
        entity_to_index = collections.OrderedDict(sorted(entity_to_index.items(), key=lambda entity_to_index: entity_to_index[1]))
        index_to_entity = [entity for i, entity in enumerate(entity_to_index)]
        logger.info("Entity vocab size: {}".format(len(entity_to_index)))
        return entity_to_index, index_to_entity
    
    def build_entity_vector(self, entity_to_index, index_to_entity, wiki2vec, idf_dict, unk_idf, en_embd_dim, entity_vector_root):
        # build entity vector from pretrained wiki vector
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
        logger.info("Saved success to {}".format(entity_vector_root))
        return entity_vector
    
    def build_entity_id(self, entity_to_index, index_to_entity, en_pad_size, entity_id_root, entity_length_root):
        # build entity index
        entity_list = self.prepare_data()[2]
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
        torch.save(all_entity_ids, entity_id_root)
        torch.save(all_entity_length, entity_length_root)
        logger.info("Saved success to {} and {}".format(entity_id_root,entity_length_root))
        return all_entity_ids, all_entity_length
    
    def build_entity_score(self, entity_freqs_file, entity_score_root):
        entity_list = self.prepare_data()[2]
        entity_score_dict = self.load_entity_score_dict(entity_freqs_file)
        all_entity_score = []
        for entities in entity_list:
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
            all_entity_score.append([score,max_score,min_score])
        all_entity_score = torch.tensor(all_entity_score)
        torch.save(all_entity_score, entity_score_root)
        logger.info("Saved success to {}".format(entity_score_root))
        return all_entity_score
    
    def build_keyword_score(self, keyword_freqs_file, keyword_score_root):
        keyword_list= self.prepare_data()[3]
        keyword_score_dict = self.load_keyword_score_dict(keyword_freqs_file)
        all_keyword_score = []
        for entities in keyword_list:
            score_list = []
            count_list = []
            for en in entities:
                if en in keyword_score_dict:
                    en_score = float(keyword_score_dict[en][0])
                    en_count = int(keyword_score_dict[en][1])
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
            all_keyword_score.append([score,max_score,min_score])
        all_keyword_score = torch.tensor(all_keyword_score)
        torch.save(all_keyword_score, keyword_score_root)
        logger.info("Saved success to {}".format(keyword_score_root))
        return all_keyword_score
    
    def build_keyword_entropy(self, keyword_entropy_file, keyword_entropy_root):
        keyword_list= self.prepare_data()[3]
        keyword_entropy_dict = self.load_keyword_entropy_dict(keyword_entropy_file)
        keyword_entropy_mean = np.mean(list(map(float,list(keyword_entropy_dict.values()))))
        all_keyword_entropy = []
        for keywords in keyword_list:
            entropy = 0
            num_words = 0
            for word in keywords:
                if word in keyword_entropy_dict:
                    num_words += 1
                    word_entropy = float(keyword_entropy_dict[word])
                    entropy += word_entropy
            if num_words == 0:
                all_keyword_entropy.append(math.exp(keyword_entropy_mean))
            else:
                all_keyword_entropy.append(math.exp(entropy/len(keywords)))
        all_keyword_entropy = torch.tensor(all_keyword_entropy)
        torch.save(all_keyword_entropy, keyword_entropy_root)
        logger.info("Saved success to {}".format(keyword_entropy_root))
        return all_keyword_entropy
    
    def build_domain_score(self, domain_freqs_file, domain_score_root):
        domain_list = self.prepare_data()[4]
        domain_score_dict = self.load_domain_score_dict(domain_freqs_file)
        all_domain_score = []
        for domain in domain_list:
            score = 1
            if domain in domain_score_dict:
                score = float(domain_score_dict[domain])
            all_domain_score.append(score)
        all_domain_score = torch.tensor(all_domain_score)
        torch.save(all_domain_score, domain_score_root)
        logger.info("Saved success to {}".format(domain_score_root))
        return all_domain_score
    
    def build_category_score(self, cat_1_freqs_file,cat_2_freqs_file,cat_3_freqs_file,cat_score_root):
        cat_1_list,cat_2_list,cat_3_list = self.prepare_data()[5]
        cat_1_dict,cat_2_dict,cat_3_dict = self.load_category_freqs_dict(cat_1_freqs_file,cat_2_freqs_file,cat_3_freqs_file)
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
        torch.save(all_cat_score, cat_score_root)
        logger.info("Saved success to {}".format(cat_score_root))
        return all_cat_score
        
    def en_vector_norm(self, vector):
        # normlize entity vector
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-9)
    
    def load_all_input_data(self, ratio, batch_size, text_id_root, labels_root,entity_id_root, entity_length_root, 
                  entity_score_root, keyword_score_root, keyword_entropy_root, domain_score_root, cat_score_root):
        # build dataset and load dataloader
        all_input_ids = torch.load(text_id_root)
        all_entity_ids = torch.load(entity_id_root)
        all_entity_length = torch.load(entity_length_root)
        all_entity_score = torch.load(entity_score_root)
        all_keyword_score = torch.load(keyword_score_root)
        all_keyword_entropy = torch.load(keyword_entropy_root)
        all_domain_score = torch.load(domain_score_root)
        all_cat_score = torch.load(cat_score_root)
        labels = torch.load(labels_root)
        # Split data into train and validation
        dataset = TensorDataset(
            all_input_ids,
            all_entity_ids,        
            all_entity_length,
            all_entity_score,
            all_keyword_score,
            all_keyword_entropy,
            all_domain_score,
            all_cat_score,
            labels
        )
        train_size = int(ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # Create train and validation dataloaders
        all_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
        
        logger.info("Num of all_dataloader: {}".format(len(all_dataloader)))
        logger.info("Num of train_dataloader: {}".format(len(train_dataloader)))
        logger.info("Num of valid_dataloader: {}".format(len(valid_dataloader)))

        return all_dataloader, train_dataloader, valid_dataloader
    
    def load_bert_input_data(self, ratio, batch_size, text_id_root, labels_root,entity_id_root, entity_length_root):
        # build dataset and load dataloader
        all_input_ids = torch.load(text_id_root)
        all_entity_ids = torch.load(entity_id_root)
        all_entity_length = torch.load(entity_length_root)
        labels = torch.load(labels_root)
        # Split data into train and validation
        dataset = TensorDataset(
            all_input_ids,
            all_entity_ids,        
            all_entity_length,
            labels
        )
        train_size = int(ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # Create train and validation dataloaders
        all_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
        
        logger.info("Num of all_dataloader: {}".format(len(all_dataloader)))
        logger.info("Num of train_dataloader: {}".format(len(train_dataloader)))
        logger.info("Num of valid_dataloader: {}".format(len(valid_dataloader)))

        return all_dataloader, train_dataloader, valid_dataloader
    
    def load_idf(self, idf_file):
        # load idf file
        ret = {}
        with open(idf_file) as f:
            for line in f:
                phrase, count, idf = line.strip().split('\t')
                idf = float(idf)
                ret[phrase] = idf
        logger.info("Load success from {}".format(idf_file))
        return ret, ret['<UNK>']

    def load_entity_score_dict(self, entity_freqs_file, min_count=10):
        entity_score_dict = {}
        with open(entity_freqs_file) as f:
            for line in f:
                entity, c1, c2, freq = line.strip().split('\t')
                c1 = int(c1)
                c2 = int(c2)
                if c1 == 0 or c2 == 0:
                    c1 += 1
                    c2 += 1
                if c1 + c2 > min_count:
                    entity_score_dict[entity] = [freq, c1+c2]
        logger.info("Entity score vocab size: {}".format(len(entity_score_dict)))
        return entity_score_dict
    
    def load_keyword_score_dict(self, keyword_freqs_file, min_count=10):
        keyword_score_dict = {}
        with open(keyword_freqs_file) as f:
            for line in f:
                keyword, c1, c2, freq = line.strip().split('\t')
                c1 = int(c1)
                c2 = int(c2)
                if c1 == 0 or c2 == 0:
                    c1 += 1
                    c2 += 1
                if c1 + c2 > min_count:
                    keyword_score_dict[keyword] = [freq, c1+c2]
        logger.info("Keyword score vocab size: {}".format(len(keyword_score_dict)))
        return keyword_score_dict
    
    def load_domain_score_dict(self, domain_freqs_file, min_count=10):
        domain_score_dict = {}
        with open(domain_freqs_file) as f:
            for line in f:
                domain, c1, c2, freq = line.strip().split('\t')
                c1 = int(c1)
                c2 = int(c2)
                if c1 == 0 or c2 == 0:
                    c1 += 1
                    c2 += 1
                if c1 + c2 > min_count:
                    domain_score_dict[domain] = freq
        logger.info("domain score vocab size: {}".format(len(domain_score_dict)))
        return domain_score_dict
    
    def load_keyword_entropy_dict(self, keyword_entropy_file):
        keyword_entropy_dict = {}
        with open(keyword_entropy_file) as f:
            for line in f:
                keyword, entropy = line.strip().split('\t')
                keyword_entropy_dict[keyword] = entropy
        logger.info("keyword entropy vocab size: {}".format(len(keyword_entropy_dict)))
        return keyword_entropy_dict
    
    def load_category_freqs_dict(self, cat_1_freqs_file,cat_2_freqs_file,cat_3_freqs_file, min_count=10):
        cat_1_dict = {}
        cat_2_dict = {}
        cat_3_dict = {}
        with open(cat_1_freqs_file) as f:
            for line in f:
                cat, c1, c2, freq = line.strip().split('\t')
                c1 = int(c1)
                c2 = int(c2)
                if c1 == 0 or c2 == 0:
                    c1 += 1
                    c2 += 1
                if c1 + c2 > min_count:
                    cat_1_dict[cat] = freq
        with open(cat_2_freqs_file) as f:
            for line in f:
                cat, c1, c2, freq = line.strip().split('\t')
                c1 = int(c1)
                c2 = int(c2)
                if c1 == 0 or c2 == 0:
                    c1 += 1
                    c2 += 1
                if c1 + c2 > min_count:
                    cat_2_dict[cat] = freq
        with open(cat_3_freqs_file) as f:
            for line in f:
                cat, c1, c2, freq = line.strip().split('\t')
                c1 = int(c1)
                c2 = int(c2)
                if c1 == 0 or c2 == 0:
                    c1 += 1
                    c2 += 1
                if c1 + c2 > min_count:
                    cat_3_dict[cat] = freq
        logger.info("cat_1 vocab size: {}".format(len(cat_1_dict)))
        logger.info("cat_2 vocab size: {}".format(len(cat_2_dict)))
        logger.info("cat_3 vocab size: {}".format(len(cat_3_dict)))
        return cat_1_dict, cat_2_dict, cat_3_dict
    
    def load_entity_vector(self, entity_vector_root):
        entity_vector = torch.load(entity_vector_root)
        logger.info("Entity vector shape: {}".format(entity_vector.shape))
        return entity_vector
    
    def load_entity_score(self, entity_score_root):
        entity_score = torch.load(entity_score_root)
        logger.info("Entity score shape: {}".format(entity_score.shape))
        return entity_score