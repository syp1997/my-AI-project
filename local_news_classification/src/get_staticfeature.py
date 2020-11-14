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
    if 'stitle' not in doc:
        doc['stitle'] = ''
    if 'seg_content' not in doc:
        doc['seg_content'] = ''

    if 'text_category_v2' in doc:
        doc['text_category'] = doc['text_category_v2']

    doc['keywords'] = {w for w in doc.get('kws', [])}
    ne_loc = set()
    ne_loc.update(doc.get('ne_content_location', {}).keys())
    ne_loc.update(doc.get('ne_title_location', {}).keys())
    ne_per = set()
    ne_per.update(doc.get('ne_content_person', {}).keys())
    ne_per.update(doc.get('ne_title_person', {}).keys())
    ne_org = set()
    ne_org.update(doc.get('ne_content_organization', {}).keys())
    ne_org.update(doc.get('ne_title_organization', {}).keys())
    doc['ne_loc'] = ne_loc
    doc['ne_per'] = ne_per
    doc['ne_org'] = ne_org
    process_dateline(doc)


def process_dateline(doc):
    content = doc['seg_content']
    for dash in [' -- ', ' - ', ' — ']:
        dash_idx = content.find(dash)
        if 0 < dash_idx < 50:
            doc['dateline'] = content[:dash_idx]
            break


