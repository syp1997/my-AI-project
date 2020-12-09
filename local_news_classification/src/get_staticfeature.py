import os
from pymongo import MongoClient

db_server_address='mongodb://172.24.25.74:27017,172.31.24.51:27017/?replicaSet=rs3'
mongo_client = MongoClient(db_server_address)
db = mongo_client.staticFeature
doc_collection = db.document

fields = {
    '_id': 1,
    'url': 1,
    'seg_title': 1,
    'seg_content': 1,
    'kws': 1,
    'domain': 1,
    'stitle': 1,
    'ne_content_location': 1,
    'ne_content_organization': 1,
    'ne_content_person': 1,
    'ne_title_location': 1,
    'ne_title_organization': 1,
    'ne_title_person': 1,
    'text_category': 1,
    'text_category_v2': 1,
}


def find_doc(docid):
    try:
        doc_it = doc_collection.find({'_id': docid}, fields)
        doc = next(doc_it)
        return doc
    except Exception:
        print('Error when finding doc', docid)
        return None


def preprocess_doc(doc):
    if not doc:
        return
    if 'seg_title' in doc:
        doc['stitle'] = doc['seg_title']

    if 'stitle' not in doc:
        doc['stitle'] = ''
        
    if 'seg_content' not in doc:
        doc['seg_content'] = ''

    doc['keywords'] = {w for w in doc.get('kws', [])}
    
    def merge_entity(en,dic):
        for k,v in dic.items():
            en[k] = en.get(k,0)+v
    entity_all = {}
    en_list = [
        'ne_content_location', 
        'ne_title_location', 
        'ne_content_organization', 
        'ne_title_organization',
        'ne_content_person',
        'ne_title_person'
    ]
    for dic in en_list:
        merge_entity(entity_all, doc.get(dic, {}))
    en_sort=sorted(entity_all.items(), key = lambda item:item[1], reverse=True)
    entity_uniq = [en[0] for en in en_sort]
    doc['entity'] = entity_uniq
    
    if 'text_category_v2' in doc:
        doc['text_category'] = doc['text_category_v2']
    cat_1 = ''
    if 'text_category' in doc and 'first_cat' in doc['text_category']:
        first_cat = doc['text_category']['first_cat']
        for k,v in first_cat.items():
            if v == max(first_cat.values()):
                cat_1  = k
                break
    cat_2 = set()
    cat_2 = (doc.get('text_category', {}).get('second_cat',''))
    cat_3 = set()
    cat_3 = (doc.get('text_category', {}).get('third_cat',''))
    doc['cat_1'] = cat_1
    doc['cat_2'] = cat_2
    doc['cat_3'] = cat_3
    
    process_dateline(doc)


def process_dateline(doc):
    content = doc['seg_content']
    for dash in [' -- ', ' - ', ' — ']:
        dash_idx = content.find(dash)
        if 0 < dash_idx < 50:
            doc['dateline'] = content[:dash_idx]
            break


