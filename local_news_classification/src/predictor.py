import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        logger.info('use device:'.format(self.device))
        
    def predict(self):
        model = self.model
        model.eval()
    
        all_y_pred = []
        pbar = tqdm(enumerate(self.test_loader),total=len(self.test_loader))
        for it, (text_ids, entity_vectors, entity_length, entity_score) in pbar:
            # place data on the correct device
            text_ids = text_ids.to(self.device)
            entity_vectors = entity_vectors.to(self.device)
            entity_length = entity_length.to(self.device)
            entity_score = entity_score.to(self.device)
            with torch.set_grad_enabled(False):
                y_pred = torch.sigmoid(model(text_ids, None, entity_length, entity_score, entity_vectors))
                all_y_pred.extend(y_pred)
            pbar.set_description(f"Test Progress")
        all_y_pred = torch.stack(all_y_pred, dim=0)
        return all_y_pred