import scipy.io as sp
import numpy as np

def get_test_data():
    name = 'data'
    tot_data = sp.loadmat(name)

    tot_data = [value for key,value in tot_data.items()];

    tot_data = np.array(tot_data);
    tot_data = np.delete(tot_data,[1,11],axis=0)
    test_data = tot_data[0:10];
    train_data=tot_data[10:];
    n_classes = 10;
    n_dims = 60;
    return(test_data,train_data)
def get_labels(Data,n_classes):
    sizes_dataelems = np.array([(np.shape(ele))[0] for ele in Data]);
    no_samples = np.sum(sizes_dataelems);
    labels = np.empty(no_samples);
    indx = 0;
    for i in range(n_classes):
        labels[indx:indx+sizes_dataelems[i]] = i*(np.ones(sizes_dataelems[i]));
        indx = indx+sizes_dataelems[i];
    return labels;








    """
    sizes_testclasses = [(np.shape(ele))[0] for ele in test_data];
    sizes_testclasses = np.array(sizes_testclasses);
    n_testeles = np.sum(sizes_testclasses);
    test_labels = np.empty(n_testeles)
    indx = 0
    for i in range(n_classes):
        test_labels[indx:indx+sizes_testclasses[i]] = i*(np.ones(sizes_testclasses[i]));
        indx = indx+sizes_testclasses[i];
    #[no_samples,no_features] = np.size(test_data);
    test_data_final = test_data[0]
    for ele in test_data[1:]:
        test_data_final = np.vstack((test_data_final,ele));
    return(test_data,test_labels)
"""
(test_data,train_data) = get_test_data();
q = get_labels(test_data,10)
print(test_data,q)
