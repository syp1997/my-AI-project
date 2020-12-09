import csv
import json
import argparse
import collections


def merge_entity(en,dic):
    for k,v in dic.items():
        en[k] = en.get(k,0)+v


if __name__=='__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-i','--input')
    aparser.add_argument('-o','--output')
    aparser.add_argument('-y','--label')
    flags = aparser.parse_args()
    
fout = open(flags.output,'w')
with open(flags.input) as f:
    num = 0
    for line in f:
        entity = []
        keywords = []
        num += 1
        if num % 10000 == 0:
            print("Process data {}".format(num))
        js = json.loads(line)
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
            merge_entity(entity_all, js.get(dic, {}))
        en_sort=sorted(entity_all.items(), key = lambda item:item[1], reverse=True)
        entity_uniq = [en[0] for en in en_sort]
        
        keywords = {w for w in js.get('kws', [])}
        
        docid = js['_id']
        text = js['stitle'] + ' . ' + js['seg_content']
        en = '|'.join(entity_uniq)
        kws = '|'.join(keywords)
        domain = js.get('domain', '')
        category = js.get('text_category', {})
        cat_1 = ''
        cat_2 = []
        cat_3 = []
        if 'first_cat' in category:
            first_cat = category['first_cat']
            for k,v in first_cat.items():
                if v == max(first_cat.values()):
                    cat_1  = k
        if 'second_cat' in category:
            cat_2.extend(category['second_cat'].keys())
        cat_2 = '|'.join(cat_2)
        if 'third_cat' in category:
            cat_3.extend(category['third_cat'].keys())
        cat_3 = '|'.join(cat_3)
        label = flags.label
        fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(docid,text,en,kws,domain,cat_1,cat_2,cat_3,label))
