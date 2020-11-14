import os
import json
import time
from collections import OrderedDict
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import torch
from transformers import BertTokenizer
from wikipedia2vec import Wikipedia2Vec
import joblib

from flask import Flask, jsonify, request
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from werkzeug.http import HTTP_STATUS_CODES

from get_staticfeature import find_doc, preprocess_doc
from data_processor import DataProcess
from model import Model, ModelConfig
from feature_extrator import FeatureExtrator
from test_data_collator import TestDataCollator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    logHandler = TimedRotatingFileHandler('logs/prod_pipeline.log', when='midnight', interval=1, backupCount=30)
    formatter = logging.Formatter('[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s')
    logHandler.setFormatter(formatter)
    gunicorn_logger.addHandler(logHandler)
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

bootstrap = Bootstrap(app)
moment = Moment(app)


def error_response(status_code, message=None):
    payload = {'error': HTTP_STATUS_CODES.get(status_code, 'Unknown error')}
    if message:
        payload['message'] = message
    response = jsonify(payload)
    response.status_code = status_code
    return response


def bad_request(message):
    return error_response(400, message)


BASE_DIR = '/mnt/nlp/serving/locallikelihood'
idf_file = os.path.join(BASE_DIR, 'idf_bigram5.txt')
entity_vector_root = os.path.join(BASE_DIR, 'entity_vectors_1028.pt')
bert_model_file = os.path.join(BASE_DIR, 'models/bert_model.pt')
xgboost_model_file = os.path.join(BASE_DIR, 'models/xgboost.model_1')
entity_frep_file = os.path.join(BASE_DIR, 'entity_frep.tsv')
domain_frep_file = os.path.join(BASE_DIR, 'domain_frep.tsv')
keyword_entropy_file = os.path.join(BASE_DIR, 'keywords_entropy.tsv')

# use pretrained bert model and tokenizer
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
# use pretrained wiki_vector model
wiki_model_file = os.path.join(BASE_DIR, 'wiki_vector.model')
wiki2vec = Wikipedia2Vec.load(wiki_model_file)
en_embd_dim = 100 # entity embedding dim
en_pad_size = 12 # max entity number of one data
device = torch.device("cpu")   # use CPU or GPU
app.logger.info('use device: {}'.format(device))

processor = DataProcess()
entity_vector = processor.load_entity_vector(entity_vector_root) # get pretrained entity_vector

mconf = ModelConfig(
    bert_model_name, 
    entity_vector, 
    en_embd_dim, 
    en_hidden_size1=128, 
    en_hidden_size2=128, 
    use_en_encoder=True
)

bert = Model(mconf).to(device)
bert.load_state_dict(torch.load(bert_model_file, map_location=torch.device('cpu'))) #load bert model
xgboost_model = joblib.load(xgboost_model_file) #load xgboost model

test_batch = 32
idf_dict, unk_idf = processor.load_idf(idf_file)
entity_score_dict = processor.load_entity_score_dict(entity_frep_file)
keyword_entropy_dict = processor.load_keyword_entropy_dict(keyword_entropy_file)
domain_score_dict = processor.load_domain_score_dict(domain_frep_file)
keyword_entropy_mean = np.mean(list(map(float,list(keyword_entropy_dict.values()))))
app.logger.info('Load all data successfully!')


@app.route('/score', methods=['GET', 'POST'])
def score():
    doc = request.json
    preprocess_doc(doc)
    try:
        url = doc['url']
        stime = time.time()
        entitiy_set = set()
        entitiy_set.update(doc['ne_loc'],doc['ne_org'],doc['ne_per'])
        entities = '|'.join(entitiy_set)
        keywords = '|'.join(doc['keywords'])
        test_data = [
            doc['_id'], 
            doc['stitle']+' . '+doc['seg_content'], 
            entities, 
            keywords, 
            doc['domain']
        ]
        test_features = get_features([test_data])
        score = xgboost_model.predict_proba(test_features.numpy())[0,1]
        score = float(score)
        etime = time.time()
        app.logger.info('{} extract success, cost: {}'.format(url, etime - stime))
    except Exception as e:
        app.logger.fatal('{} extract fail'.format(url))
        app.logger.exception(e)

    result = json.dumps({
        'local_socre': score,
    })
    
    return result


@app.route('/debug', methods=['GET'])
def debug():
    docid = request.args['docid']
    app.logger.info('Processing {}'.format(docid))
    doc = find_doc(docid)
    preprocess_doc(doc)
    if not doc:
        return json.dumps('Doc not found: ' + docid)
    res = OrderedDict()
    result = OrderedDict()
    result['docid'] = docid
    try:
        url = doc['url']
        stime = time.time()
        entitiy_set = set()
        entitiy_set.update(doc['ne_loc'],doc['ne_org'],doc['ne_per'])
        entities = '|'.join(entitiy_set)
        keywords = '|'.join(doc['keywords'])
        test_data = [
            doc['_id'], 
            doc['stitle']+' . '+doc['seg_content'], 
            entities, 
            keywords, 
            doc['domain']
        ]
        test_features = get_features([test_data])
        score = xgboost_model.predict_proba(test_features.numpy())[0,1]
        score = float(score)
        result['title'] = doc['stitle']
        result['domain'] =doc['domain']
        result['url'] = url
        result['local_score'] = score
        result['entity'] = entities
        result['keywords'] = keywords.replace('^^',' ')
        result['features'] = [
            {'key':'bert score','value':test_features[0,0].item()},
            {'key':'entity score','value':test_features[0,1].item()},
            {'key':'keywords entropy','value':test_features[0,2].item()},
        ]
        etime = time.time()
        app.logger.info('{} extract success, cost: {}'.format(url, etime - stime))
    except Exception as e:
        app.logger.fatal('{} extract fail'.format(url))
        app.logger.exception(e)
        
    res['result'] = result
    results = json.dumps(res, default=str)

    return results


def get_features(test_data):
    test_data_collator = TestDataCollator(test_data, None)
    test_dataloader = test_data_collator.load_data(
        test_batch, tokenizer, wiki2vec, idf_dict, unk_idf, 
        en_pad_size, en_embd_dim, entity_score_dict, 
        keyword_entropy_dict, domain_score_dict, keyword_entropy_mean
    )
    test_feature_extrator = FeatureExtrator(bert, test_dataloader, device)
    bert_score_test, entity_score_test, keyword_entropy_test, domain_score_test = test_feature_extrator.get_features(test=True)
    test_features = test_feature_extrator.process_features(
        bert_score_test, entity_score_test, 
        keyword_entropy_test, domain_score_test
    )
    return test_features


if __name__ == '__main__':
    
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    app.run(debug=False, host='0.0.0.0', port=8923)

