import pandas as pd
import numpy as np
# from sklearn.metrics import pairwise_distances

datasets = ['uber-raw-data-apr14.csv',
            'uber-raw-data-aug14.csv',
            'uber-raw-data-jul14.csv',
            'uber-raw-data-jun14.csv',
            'uber-raw-data-may14.csv',
            'uber-raw-data-sep14.csv']

print(datasets)
print()

total_data = pd.DataFrame(columns=['Date/Time', 'Lat', 'Lon', 'Base'])
print(total_data)
print(total_data.shape)

for dataset in datasets:
    data = pd.read_csv('uber_archive/'+dataset, header=0, delimiter=',')  # uber - covertype
    #print(data)
    print(data.shape)
    total_data = pd.concat([total_data, data])
    #print(total_data)
    #print(total_data.shape)
    #print()

print(total_data)
print(total_data.shape)
print()

cols = total_data.shape[1]
#print(cols)


S = np.matrix(total_data.iloc[:, 1:cols - 1].values)  # 2 var
print(S)
print(S.shape)
print()

max_S = np.max(S[:,:],axis=0)
min_S = np.min(S[:,:],axis=0)
diff_S = max_S - min_S

print(max_S)
print(min_S)
print(diff_S)
print()

OUTPUT = np.divide(np.subtract(S[:, ], min_S), diff_S)
print(OUTPUT)
print(OUTPUT.shape)
print()

pd.DataFrame(OUTPUT).to_csv("uber_output_norm.csv", header=['Lat', 'Lon'], index=None)
#np.savetxt("uber_output.csv", OUTPUT, delimiter=",")

