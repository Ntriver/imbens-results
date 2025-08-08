import numpy as np
import os
from sklearn.model_selection import RepeatedStratifiedKFold

print(os.listdir('./data/processed_data/'))
# npzfile = np.load('./data/processed_data/ecoli.npz')

dataset_root = './data/processed_data/'
# print(os.path.join(dataset_root,'ecoli.npz'))
all_file_names = os.listdir(dataset_root)

# file_name = all_file_names[0]
# count=0
# print('./data/5T5CV/'+file_name[:-4]+'/'+file_name[:-4]+'-'+str(count)+'.npz')
# exit()

for file_name in all_file_names:
    npzfile = np.load(os.path.join(dataset_root,file_name))
    data, target = npzfile['data'], npzfile['target']

    skf = RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=20252025)

    for i, (train_index, test_index) in enumerate(skf.split(data, target)):
        print(f"fold {i}:")
        x_train = data[train_index]
        y_train = target[train_index]
        x_test = data[test_index]
        y_test = target[test_index]
        saved_path = './data/10T5CV/'+file_name[:-4]+'/'
        if not os.path.isdir(saved_path):
            os.mkdir(saved_path)
        np.savez(saved_path+file_name[:-4]+'-'+str(i)+'.npz',x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
    # exit()

