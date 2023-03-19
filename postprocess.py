import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,classification_report,confusion_matrix

def postprocess(preds,size):

    results = preds.copy()
    i=0
    while i + size <len(preds):
        counts = np.bincount(preds[i:i+size])
        mode = np.argmax(counts)
        results[i:i+size] = np.repeat(mode, size)
        i+=size
        # results.extend(result)
        
    return results

def postprocess_overlap(preds,size):

    overlap = int(size/2)
    results = preds.copy()
    i=0
    while i + size <len(preds):
        counts = np.bincount(preds[i:i+size])
        mode = np.argmax(counts)
        results[i:i+size] = np.repeat(mode, size)
        i+=overlap
        # i+=int(overlap/2 + overlap)
        
    return results

def postprocess_smooth(preds,size):

    results = preds.copy()
    len_preds = len(preds)
    # l = 
    for i in range(len_preds):
        if i< size:
            continue
        elif i>len_preds-size:
            continue
        else:
            counts = np.bincount(preds[i-size:i+size])
            mode = np.argmax(counts)
            results[i] = mode
    

    return results


def find_best(preds,val,size):
    # preds_best = postprocess(preds,size)
    preds_best = postprocess_overlap(preds,size)
    # preds_best = postprocess_smooth(preds,size)
    report = classification_report(preds_best, val, digits=4)
    C = confusion_matrix(preds_best, val)
    score = f1_score(val, preds_best, average='macro')

    return score

if __name__ == '__main__':
    preds = pd.read_csv('output/vit_ms1m_5_lr5/preds.csv')

    val = pd.read_csv('data/val.csv')
    preds = preds.label.values
    val=val.label.values
    assert len(preds)==len(val), 'invalid preds'
    # size_range = range(2,50)
    best = 0
    size_best = 0
    # size1 = 10
    # a = find_best(preds,val,200)
    # print(a)
    # a = find_best(preds,val,300)
    # print(a)
    a = find_best(preds,val,180)
    print(a)


    for size in range(4,1000,2):
        print(size)
        score = find_best(preds,val,size)
        if score > best:
            best = score
            size_best = size
            print(f"The best window size is {size_best}, The best f1score is {best}")
    print(f"The best window size is {size_best}, The best f1score is {best}")