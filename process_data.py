from imbalanced_ensemble.datasets import fetch_datasets
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import numpy as np



dataset = fetch_datasets(data_home='./data',filter_data=range(1,28),verbose=True)
keys = [key for key in dataset.keys()]
for i in range(0,27):
    print('dataset #%d, name: %s' % (i,keys[i]))
    data, target = dataset[keys[i]]['data'], dataset[keys[i]]['target']
    print(data.shape, sorted(Counter(target).items()))
    print()

    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    # print(data.min())

    np.savez('./data/processed_data/'+keys[i]+'.npz', data=data, target=target)
# npzfile = np.load('./data/processed_data/ecoli.npz')
# print(npzfile['a'].shape)
# print(npzfile['b'].shape)


# print(dataset.keys())
# print(dataset['ecoli']['data'],dataset['ecoli']['target'])