import scipy.io as sp
import numpy as np

def get_test_data():
    name = 'data'
    tot_data = sp.loadmat(name)
    sorted_labels = sorted(tot_data);
    test_data = np.array([tot_data[label] for label  in sorted_labels[3:13]]);
    train_data = np.array([tot_data[label] for label in sorted_labels[13:23]]);
    n_classes = 10;
    n_dims = 60;
    return(test_data,train_data)

def vert_cat(Data,n_classes,n_dims):
    alldata_samples = Data[0];
    for i in range(1,n_classes):
        alldata_samples = np.vstack((alldata_samples,Data[i]));
    return alldata_samples

def get_labels(Data,n_classes):
    sizes_dataelems = np.array([(np.shape(ele))[0] for ele in Data]);
    no_samples = np.sum(sizes_dataelems);
    labels = np.empty((no_samples,1));
    indx = 0;
    for i in range(n_classes):
        labels[indx:indx+sizes_dataelems[i],0] = i*(np.ones(sizes_dataelems[i]));
        indx = indx+sizes_dataelems[i];
    return (labels,sizes_dataelems);

def receive_data():
    (test_data,train_data) = get_test_data()

    (test_labels,test_class_sizes) = get_labels(test_data,10);
    (train_labels,test_class_sizes) = get_labels(train_data,10);

    all_test_data = vert_cat(test_data,10,60);
    all_train_data = vert_cat(train_data,10,60);

    all_test_data = np.hstack((all_test_data,test_labels));
    all_train_data = np.hstack((all_test_data,test_labels));

    return((all_train_data,all_test_data))






