# -*- coding:utf-8 -*-
import pickle
import os
import random
import glob

root='./data'
classes={0:'/adidas/', 1:'/converse/',2:'/nike/'}
meta_dict={}
train_id=[]
for key in classes:
    list_dir=glob.glob(root+'/train'+classes[key]+'*.jpg')
    for name in list_dir:
            meta_dict[name] = key
    train_id+=list_dir

random.shuffle(train_id)
K=int(len(train_id)/10)
train=train_id[:9*K]
valid=train_id[9*K:]

random.shuffle(train)
random.shuffle(valid)

print(len(train),len(valid))

with open(root+'/train_labels.pkl', 'wb') as f:
    pickle.dump(meta_dict, f)
    f.close()

with open(root+'/train_valid_id.pkl', 'wb') as f:
    pickle.dump([train,valid], f)
    f.close()


###################################### test ############################
meta_dict={}
test_id=[]
for key in classes:
    list_dir=glob.glob(root+'test/'+classes[key]+'*.jpg')
    for name in list_dir:
            meta_dict[name] = key
    test_id+=list_dir
print(len(test_id))

with open(root+'/test_labels.pkl', 'wb') as f:
    pickle.dump(meta_dict, f)
    f.close()

with open(root+'/test_id.pkl', 'wb') as f:
    pickle.dump([test_id], f)
    f.close()
