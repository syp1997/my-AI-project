import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FeatureExtrator:

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        logger.info('use device: {}'.format(self.device))
        
    def generate_score(self, bert_score_root, test=False):
        model = self.model
        model.eval()
    
        all_y_pred = []
        pbar = tqdm(enumerate(self.data_loader),total=len(self.data_loader))
        for it, data in pbar:
            if test:
                text_ids, entity_vectors, entity_length = data[:3]
                entity_vectors = entity_vectors.to(self.device)
                entity_ids = None
            else:
                text_ids, entity_ids, entity_length = data[:3]
                entity_ids = entity_ids.to(self.device)
                entity_vectors = None
            # place data on the correct device
            text_ids = text_ids.to(self.device)
            entity_length = entity_length.to(self.device)
            with torch.set_grad_enabled(False):
                y_pred = torch.sigmoid(model(text_ids, entity_ids, entity_length, entity_vectors))
                all_y_pred.extend(y_pred)
            pbar.set_description(f"Test Progress")
        all_y_pred = torch.stack(all_y_pred, dim=0)
        torch.save(all_y_pred, bert_score_root)
        logger.info("Saved success to {}".format(bert_score_root))
        return all_y_pred
        
    def get_features(self, bert_score_root):
        
        bert_score = torch.load(bert_score_root)
        all_entity_score = []
        all_domain_score = []
        for it, data in enumerate(self.data_loader):
            text_ids, entity_vectors, entity_length, entity_score, domain_score = data[:5]
            all_entity_score.extend(entity_score)
            all_domain_score.extend(domain_score)
        all_entity_score = torch.stack(all_entity_score, dim=0)
        all_domain_score = torch.stack(all_domain_score, dim=0)
        return bert_score, all_entity_score, all_domain_score
        
    def get_labels(self):
        
        all_labels = []
        for it, data in enumerate(self.data_loader):
            y = data[-1]
            all_labels.extend(y)
        all_labels = torch.stack(all_labels, dim=0)
        return all_labels
        