import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class EntityScorer:

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        logger.info('use device: {}'.format(self.device))
        
    def generate_score(self, test = False):
        model = self.model
        model.eval()
    
        all_y_pred = []
        all_entity_distribute = []
        all_labels = []
        pbar = tqdm(enumerate(self.data_loader),total=len(self.data_loader))
        if test:
            for it, (_, entity_vectors, entity_length, entity_distribute) in pbar:
                # place data on the correct device
                entity_vectors = entity_vectors.to(self.device)
                entity_length = entity_length.to(self.device)
                with torch.set_grad_enabled(False):
                    y_pred = torch.sigmoid(model(None, entity_length, entity_vectors))
                    all_y_pred.extend(y_pred)
                    all_entity_distribute.extend(entity_distribute)
                pbar.set_description(f"Entity Model Score Progress")      
            entity_score = torch.stack(all_y_pred, dim=0)
            entity_distribute = torch.stack(all_entity_distribute, dim=0)
            return entity_score, entity_distribute
        else:
            for it, (_, entity_ids, entity_length, entity_distribute, y) in pbar:
                # place data on the correct device
                entity_ids = entity_ids.to(self.device)
                entity_length = entity_length.to(self.device)
                with torch.set_grad_enabled(False):
                    y_pred = torch.sigmoid(model(entity_ids, entity_length))
                    all_y_pred.extend(y_pred)
                    all_entity_distribute.extend(entity_distribute)
                    all_labels.extend(y)
                pbar.set_description(f"Entity Model Score Progress")
            entity_score = torch.stack(all_y_pred, dim=0)
            entity_distribute = torch.stack(all_entity_distribute, dim=0)
            all_labels = torch.stack(all_labels, dim=0)
            return entity_score, entity_distribute, all_labels