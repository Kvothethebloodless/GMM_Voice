import numpy as np


def parse_iris():
        data_newlinesplit = [st.split(',') for st in open('IRIS.txt').readlines()]
        data_newlinesplit = data_newlinesplit[0:150]
        for i in range(150):
               
                if data_newlinesplit[i][4] == 'Iris-setosa\n':
                        data_newlinesplit[i][4] = 0
                elif data_newlinesplit[i][4] == 'Iris-versicolor\n':
                        data_newlinesplit[i][4] = 1
                elif data_newlinesplit[i][4] == 'Iris-virginica\n':
                        data_newlinesplit[i][4] = 2
                data_newlinesplit[i] = [float(j) for j in data_newlinesplit[i]]
                data_newlinesplit[i] = np.array(data_newlinesplit[i]);
                #data_newlinesplit[i] = data_newlinesplit[i].astype(np.float);
                
        data_np= np.array(data_newlinesplit);
        return data_np
def get_data():
        iris_data = parse_iris();
        locs = np.arange(40)
        locs2 = np.arange(10)
        trainelems_loc= np.concatenate([locs,locs+50,locs+100])
        testelems_loc = np.concatenate([locs2+40,locs2+90,locs2+140])
        iris_traindata = iris_data[trainelems_loc]
        iris_testdata = iris_data[testelems_loc]
        return([iris_traindata,iris_testdata])   

   
