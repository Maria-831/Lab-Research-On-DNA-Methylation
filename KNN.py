import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
def KNN(data, methylation, val, k):
    predictions = []
    for index in range(len(val)):
        if index % 1000 == 0:
            print('Searching for index: ', index)
        neighbour_list = []
        neighbour_index = []
        for train_index in range(len(data)):
            diff = val[index] - data[train_index]
            score = diff.dot(diff.T)
            if len(neighbour_list) == k:
                if score < min(neighbour_list):
                    index = neighbour_list.index(min(neighbour_list))
                    neighbour_list[index] = score
                    neighbour_index[index] = train_index
            else:
                neighbour_list.append(score)
                neighbour_index.append(train_index)
        s = 0
        for i in neighbour_index:
            s += methylation[i]
        predictions.append(s/k)
    return predictions
def main():
    counts = pd.read_csv('../data/Kmers_K6_counts.csv', header = None)
    counts = counts.drop([0],axis = 1)
    CG_attributes = pd.read_csv('../data/CpG_attributes.csv')
    counts = pd.concat([counts,CG_attributes],axis = 1)
    #TF_binding = pd.read_csv('PWM_TFbinding.csv')
    #counts = pd.concat([counts,TF_binding],axis = 1)
    
    methy = pd.read_csv('../data/Mouse_DMRs_methylation_level.csv',header=None)
    
    counts = counts.as_matrix()
    methy  = methy.as_matrix()
    
    Results = []
    print('Starts Training')
    for cell_type in range(16):
        #predictions1 = KNN(counts[:48000],methy[:48000,cell_type],counts[48000:],k = 1)
        #r1 = r2_score(methy[48000:,cell_type],np.array(predictions1))
        #print('cell_type: ',cell_type, '\nscore: ', r1, 'neighbours: ', 1)
        #predictions2 = KNN(counts[:48000],methy[:48000,cell_type],counts[48000:],k = 5)
        #r2 = r2_score(methy[48000:,cell_type],np.array(predictions2))
        #print('cell_type: ',cell_type, '\nscore: ', r2, 'neighbours: ', 5)
        #predictions3 = KNN(counts[:48000],methy[:48000,cell_type],counts[48000:],k = 10)
        #r3 = r2_score(methy[48000:,cell_type],np.array(predictions3))
        #print('cell_type: ',cell_type, '\nscore: ', r3, 'neighbours: ', 10)
        predictions4 = KNN(counts[:48000],methy[:48000,cell_type],counts[48000:],k = 20)
        r4 = r2_score(methy[48000:,cell_type],np.array(predictions4))
        print('cell_type: ',cell_type, '\nscore: ', r4, 'neighbours: ', 20)
        results = [r4]
        Results.append(results)
    csv = pd.DataFrame(data = np.array(Results))
    csv.to_csv('Results_KNN_Kmers6_CpG_k20.csv',index = False)
main()
