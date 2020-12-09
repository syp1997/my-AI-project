import torch
import logging
import time

logger = logging.getLogger(__name__)


class FeatureExtrator:
    """ extract features for training xgboost"""

    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
    def generate_score(self, test):
        model = self.model
        model.eval()
    
        all_y_pred = []
        for it, data in enumerate(self.data_loader):
            entity_ids = None
            entity_vectors = None
            if test:
                text_ids, entity_vectors, entity_length = data[:3]
                entity_vectors = entity_vectors.to(self.device)
            else:
                text_ids, entity_ids, entity_length = data[:3]
                entity_ids = entity_ids.to(self.device)
            # place data on the correct device
            text_ids = text_ids.to(self.device)
            entity_length = entity_length.to(self.device)
            with torch.set_grad_enabled(False):
                y_pred = torch.sigmoid(model(text_ids, entity_ids, entity_length, entity_vectors))
                all_y_pred.extend(y_pred)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        return all_y_pred
        
    def get_features(self, test=False):
        last_time = time.time()
        bert_score = self.generate_score(test)
        logger.info('Bert score: Took {} seconds'.format(time.time() - last_time))
        all_entity_score = []
        all_keyword_score = []
        all_keyword_entropy = []
        all_domain_score = []
        all_cat_score = []
        for it, data in enumerate(self.data_loader):
            text_ids, entity_vectors, entity_length, entity_score, keyword_score, keyword_entropy, domain_score, cat_score = data[:8]
            all_entity_score.extend(entity_score)
            all_keyword_score.extend(keyword_score)
            all_keyword_entropy.extend(keyword_entropy)
            all_domain_score.extend(domain_score)
            all_cat_score.extend(cat_score)
        all_entity_score = torch.stack(all_entity_score, dim=0)
        all_keyword_score = torch.stack(all_keyword_score, dim=0)
        all_keyword_entropy = torch.stack(all_keyword_entropy, dim=0)
        all_domain_score = torch.stack(all_domain_score, dim=0)
        all_cat_score = torch.stack(all_cat_score, dim=0)
        return bert_score, all_entity_score, all_keyword_score, all_keyword_entropy, all_domain_score, all_cat_score
        
    def get_labels(self):
        all_labels = []
        for it, data in enumerate(self.data_loader):
            y = data[-1]
            all_labels.extend(y)
        all_labels = torch.stack(all_labels, dim=0)
        return all_labels
    
    def process_features(self, bert_score, entity_score, keyword_score, keyword_entropy, domain_score, cat_score):
        m = bert_score.shape[0]
        bert_score = bert_score.to('cpu').view(m,-1)
        entity_score = entity_score.view(m,-1)
        keyword_score = keyword_score.view(m,-1)
        keyword_entropy = keyword_entropy.view(m,-1)
        domain_score = domain_score.view(m,-1)
        cat_score = cat_score
        features = torch.cat([bert_score, entity_score, keyword_score, keyword_entropy, cat_score],dim=1)
        return features
    
    def process_labels(self, labels):
        return labels.view(labels.shape[0],-1)
    
    def load_train_features(self, bert_score_root, entity_score_root, keyword_score_root, 
                            keyword_entropy_root, domain_score_root, category_score_root, labels_root):
        bert_score = torch.load(bert_score_root).to('cpu')
        m = bert_score.shape[0]
        bert_score = bert_score.view(m,-1)
        entity_score = torch.load(entity_score_root)[:m].view(m,-1)
        keyword_score = torch.load(keyword_score_root)[:m].view(m,-1)
        keyword_entropy = torch.load(keyword_entropy_root)[:m].view(m,-1)
        domain_score = torch.load(domain_score_root)[:m].view(m,-1)
        cat_score = torch.load(category_score_root)[:m].view(m,-1)
        features = torch.cat([bert_score, entity_score, keyword_score, keyword_entropy, cat_score],dim=1)
        labels = torch.load(labels_root)[:m].view(m,-1)
        print("Features shape: {}".format(features.shape))
        return features, labels
        