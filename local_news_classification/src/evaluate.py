import argparse
import os 
import json
import csv
import requests
from multiprocessing import Pool
import time
import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
BASE_DIR = '/data/suyinpei/'
server_address = None


def request(docid):
    if not docid:
        print('docid empty, return.')
        return -1
    r = requests.get(server_address, params={'docid': docid})
    try:
        js = r.json()
    except Exception as e:
        logger.exception(e)
        return -1
    result = js['result']
    local_score = result['local_score']
    return local_score


def get_docid(line):
    line = line.strip()
    if len(line) == 8:
        return line
    if len(line.split('\t')[0]) == 8:
        return line.split('\t')[0]
    try:
        js = json.loads(line)
        if '_id' in js:
            return js['_id']
        elif 'docid' in js:
            return js['docid']
        else:
            return None
    except Exception:
        return None
    
    
def evaluate(gold_file, y_pred, threshold):
    all_num, tp, fp, fn, tn = 0, 0, 0, 0, 0
    f = open(gold_file)
    reader = csv.reader(f, delimiter = '\t')
    for row, pred in zip(reader,y_pred):
        if row[-1] != '0' and row[-1] != '1':
            continue
        else:
            label = float(row[-1])
            all_num += 1
        score = float(pred >= threshold)
        if label == 1 and score == 1:
            tp += 1
            a = 1
        elif label == 0 and score == 1:
            fp += 1
            a = 0
        elif label == 1 and score == 0:
            fn += 1
            a = 0
        elif label == 0 and score == 0:
            tn += 1
            a = 1
        else:
            print("no match!!")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    logger.info(
        "all num: {}\ntrue positive: {}\nfalse positive: {}\nfalse negtive: {}\ntrue negtive: {}\nprecision: {}\nrecall: {}\nF-Score:{}\n".format(all_num, tp, fp, fn, tn, tp/(tp+fp),tp/(tp+fn),(2*(tp/(tp+fp))*tp/(tp+fn))/(tp/(tp+fp)+tp/(tp+fn)))
    )
    return precision, recall
        
        
def plot_precision_recall_curve(l,p,r):
    plt.figure(figsize = (10,6)) 
    plt.title('Precision/Recall Curve',fontsize=20)# give plot a title
    # plt.axis([-0.1,1.1,0.5,1.1]) 
    x_smooth = np.linspace(l.min(), l.max(), 300)
    y_smooth1 = make_interp_spline(l, p)(x_smooth)
    y_smooth2 = make_interp_spline(l, r)(x_smooth)
    plt.plot(x_smooth,y_smooth1,linestyle='-', color='#2b83ba', linewidth = 2.5) 
    plt.plot(x_smooth,y_smooth2,linestyle='--',color='g', linewidth = 2.5) 
    plt.xlabel('Threshold',fontsize=18)
    plt.legend(('Precision','Recall'),frameon=False, loc='upper center',ncol=4,handlelength=4,fontsize=16) # 图例
    # plt.legend(['precision','recall'],fontsize=16,shadow=True,loc='lower right')
    plt.grid(linestyle="--", alpha=0.2) # 网格线
    plt.savefig("Precision-Recall Curve", dpi=600, bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-s', '--server_address',
                         default='http://172.31.23.186:8923/debug',
                         help='http://172.31.23.186:8923/debug')
    aparser.add_argument('-i', '--input', 
                         default=os.path.join(BASE_DIR, 'test_data_1k.tsv'),
                         help='one doc perline (docid | json)')
    aparser.add_argument('-g', '--gold_file', 
                         default=os.path.join(BASE_DIR, 'test_data_1k.tsv'),
                         help='one doc perline(docid + label)')
    aparser.add_argument('-o', '--output')
    aparser.add_argument('--threshold', default=0.5, type=float)
    aparser.add_argument('-p', '--nprocess', default=10, type=int)
    flags = aparser.parse_args()
    server_address = flags.server_address

    with open(flags.input) as fin:
        lines = [line.strip() for line in fin]
    docids = [get_docid(line) for line in lines]
    stime = time.time()
    logger.info('start evaluate: {}'.format(datetime.datetime.utcnow()))
    y_pred = []
    n = 0
    with Pool(flags.nprocess) as p:
        with open(flags.output, 'w') as fout:
            for score in p.imap(request, docids, chunksize=10):
                n += 1
                if n % 1000 == 0:
                    print('process data: ',n/10000,'w')
                y_pred.append(score)
                fout.write(str(score) + '\n')
    p,r = evaluate(flags.gold_file, y_pred, flags.threshold)
    etime = time.time()
    logger.info('evaluate {} data finished, cost: {}'.format(len(y_pred), etime - stime))

