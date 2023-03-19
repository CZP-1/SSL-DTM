import pandas as pd
import numpy as np
import os

preds_test = pd.read_csv('output/new_test_419_efficientb7/preds.csv')

preds = preds_test.label.values

# Post-process

from postprocess import *
preds_post = postprocess_overlap(preds,200)
preds_test.label = preds_post

preds_test.to_csv('output/test_fusion_200/preds_pp.csv',index=0)
preds_test['path'] = preds_test['path'].apply(lambda x: '/'.join(x.split('/')[-2:]))


# Create dictionary of image labels
preds_test = dict(zip(preds_test['path'], preds_test['label']))

with open('test_data/CVPR_5th_ABAW_EXPR_test_set_example.txt','r') as file:
    examples=file.readlines()
with open('output/test_fusion_200/predictions.txt','w') as f:
    for i in range(len(examples)):
        if i == 0:
            f.writelines(examples[0])
        elif os.path.exists('data/crop_aligned_images/cropped_aligned/'+examples[i].split(',')[0]):
            path = examples[i].split(',')[0]
            label = preds_test[path]
            f.writelines(path+','+str(label)+'\n')
        else:
            f.writelines(examples[i].split(',')[0]+','+str(label)+'\n')

with open('output/test_fusion_200/predictions.txt','r') as f1:
    submit = f1.readlines()

assert submit[0]==examples[0], "wrong predictions!"

for exp ,sub in zip (examples[1:],submit[1:]):
    assert exp.split(',')[0]==sub.split(',')[0], "wrong predictions!"









