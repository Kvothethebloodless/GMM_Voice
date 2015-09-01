import numpy as np
import matplotlib.pyplot as plt #from scipy.stats
#import multivariate_normal
import parse_iris as pi
import math as math
import scipy.io as sp
#3 important functions.

# p(c_i/X) = (p(X/c_i)*p(c_i))/p(X)
# Posterior = Likelihood * Prior/Scaling factor


#inference_likelihood /Generates the multivariate gaussian describing likelihood function of each class
#inference_prior /Calculates Prior of each class
#decision_posterior /Calculates Posterior probability of test sample and decides label assignment


def norm_pdf_multivariate(x, param):

    [size] = np.shape(x)
    pi = 3.14;
    [mu,sigma] = param;
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu);
        inv = np.linalg.inv(sigma);
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the inputs don't match")


def inference_prior(traindata):
    C_is = np.array([0,1,2])
    freq_Cis = np.array([np.sum(traindata[:,4]==C_i)  for C_i in C_is])
    prior_Cis =  freq_Cis/(np.sum(freq_Cis))
    return np.array(prior_Cis)


def decision_posterior(priors,Ci_params,datapoints,n_classes,n_datapoints,w,n_m):
    #Ci_params is a tuple with (class_labels,mu_array,var_matrix) as elements
    # Given n dimension data with I classes - size of Ci_params is 3 by
    #given data point, calculate the posterior probabilities
    #and put it back with [Ci,posterior]
    #Return Ci with maximum posterior probability.
    (class_labels,mu_array,var_2matrix) = Ci_params;
    #mu_matrix n_classes by m by n_dims matrix
    #var_matrix n_classes by m by n_dims by n_dims matrix
    posterior_prob = np.zeros((n_classes,2));
    posterior_prob[:,0] = class_labels; #Writing classes
    decided_class=np.arange(n_datapoints);
    for j in range(n_datapoints):
        datapoint =datapoints[j];
        for i in range(n_classes):
            #posterior_prob[i,1] = norm_pdf_multivariate(datapoint,(mu_array[i],var_matrix[i])) * priors[i] ;
            posterior_prob[i,1] = GMM_prob_class(datapoint,(mu_array[i],var_2matrix[i]),w[i],n_m);
        max_post_prob_loc = np.argmax(posterior_prob[:,1]);
        max_post_class = posterior_prob[:,0][max_post_prob_loc]
        decided_class[j] = max_post_class;
    return decided_class

def GMM_prob_class(x,params_class,w,n_gms): #prob = sum(wi*N(mu,sigma))
    #mu_array 1 by m mu vectors
    #var_array 1 by m var matrices
    (mu_array,var_array) = params_class
    sum = 0
    for i in range(n_gms):
        sum = sum+w[i]*norm_pdf_multivariate(x,[mu_array[i],var_array[i]]);
    return sum


def gen_respnm_expstep(classdata,mu_vector,var_matrix,w,n_dims,n_datapoints,n_features,n_m):
    respnm = np.zeros((n_datapoints,n_m))
    for i in range(n_datapoints):
        datapoint = classdata[i][0:4];
        p = GMM_prob_class(datapoint,(mu_vector,var_matrix),w,n_m);
        for j in range(n_m):
            respnm[i,j] = w[j]*norm_pdf_multivariate(datapoint,[mu_vector[j],var_matrix[j]]) / p ;
    return respnm

def gen_mu_vector_maxstep(classdata,N_M,respm,n_dims,n_datapoints):
    #x_n matrix of datapoints
    #respkm_n vector of responsibilities of each of n points for m the gaussian of all datapoints

    mu_vector = np.empty(n_dims);
    for i in range(n_dims):
        mu_vector[i]=np.sum(np.multiply(respm,classdata[:,i]));

    mu_vector = mu_vector/N_M;
    return mu_vector

def gen_var_matrix_maxstep(classdata,mu,N_M,respm,n_dims,n_datapoints): #m-dimensional data - x - n by d. mu 1 by d . Assuming dimensional independence
    var_dim = np.empty(n_dims) #Vector to hold all the important self variances i.e cov(i,j)
    for j in range(n_dims):
        var_dim[j] = np.sum(np.multiply(respm,np.power((classdata[:,j]-mu[j]),2)));
    var_matrix = np.diag(var_dim);
    var_matrix = var_matrix/N_M;
    return var_matrix

def gen_N_m(respnm,n_datapoints,n_m):
    N_m = np.empty(n_m);
    for m in range(n_m):
        N_m[m] = np.sum(respnm[:,m]);
    return N_m

def loglikelihood_error(oldparams,classdata,newparams,n_m,n_datapoints):
    l1 = log_likelihood(oldparams,classdata,n_m,n_datapoints);
    l2 = log_likelihood(newparams,classdata,n_m,n_datapoints);
    error = np.abs(l1-l2);
    return error

def log_likelihood(params,classdata,n_m,n_datapoints):
    ll = 0;
    (mu_vector,var_matrix,w) = params;
    s=0;
    for i in range(n_datapoints):
        s = s+np.log(GMM_prob_class(classdata[i],(mu_vector,var_matrix),w,n_m))
    return s

def gen_mu_vector_matrix_maxstep(dataci,respnm,N_m,n_dims,n_m,n_classes):
    mu_matrix = np.zeros((n_m,n_dims));

    for j in range(n_m):
        mu_matrix[j]= gen_mu_vector_maxstep(dataci,N_m[j],respnm[:,j],n_dims,n_datapoints);

    return mu_matrix

def gen_var_matrix_matrix_maxstep(dataci,mu_matrix,respnm,N_m,n_dims,n_m,n_classes):
    var_matrix = np.zeros((n_m,n_dims,n_dims));
    for j in range(n_m):
        var_matrix[j] = gen_var_matrix_maxstep(dataci,mu_matrix[j],N_m[j],respnm[:,j],n_dims,n_datapoints);
    return var_matrix

def gen_weight_array(N_m,n_datapoints):
    return(N_m/n_datapoints)


def gen_random_respnm(n_datapoints,n_m):
    respnm = np.empty((n_datapoints,n_m))
    for i in range(n_datapoints):
        respnm[i] = np.random.rand(n_m);
        respnm[i] = np.divide(respnm[i],np.sum(respnm[i]));
    return respnm


def get_test_data():
    name = 'data'
    tot_data = sp.loadmat(name)

    tot_data = [value for key,value in tot_data.items()];

    tot_data = np.array(tot_data);
    test_data = tot_data[0:10];
    n_classes = 10;
    n_dims = 60;
    sizes_testclasses = [(np.shape(ele))[0] for ele in test_data];
    sizes_testclasses = np.array(sizes_testclasses);
    n_testeles = np.sum(sizes_testclasses);
    test_labels = np.empty(n_testeles)
    indx = 0
    for i in range(n_classes):
        test_labels[indx:indx+sizes_testclasses[i]] = i*(np.ones(sizes_testclasses[i]));
        indx = indx+sizes_classes[i];
    #[no_samples,no_features] = np.size(test_data);
    test_data = np.reshape(test_data,(no_samples,n_dims));
    return(test_data,test_labels)



def inference_GMM(traindata,class_labels,n_classes,n_features,n_datapoints):
    n_m = 2;
    n_dim = n_features;
    #respnm=np.random.rand((n_datapoints,n_m))
    respnm = gen_random_respnm(n_datapoints,n_m) ##NOT RANDOM
    bound = 1;
    N_m = gen_N_m(respnm,n_datapoints,n_m)
    q = 0;
    logl2 = 0;
    all_ci_mu_matrix  = np.zeros((n_classes,n_m,n_dim));
    all_ci_var_matrix = np.zeros((n_classes,n_m,n_dim,n_dim));
    all_ci_weight_matrix = np.zeros((n_classes,n_m))
    for i in range(n_classes):
        data_ci = traindata[np.where((traindata[:,4]==class_labels[i]))];
        data_ci = data_ci[:,0:4];
        error = 9999999999999;
        n=0
        logl=[];

        logl2 = 0;
        while(error>.0001):

            mu_matrix = gen_mu_vector_matrix_maxstep(data_ci,respnm,N_m,n_features,n_m,n_classes);
            #print(mu_matrix)
            var_matrix = gen_var_matrix_matrix_maxstep(data_ci,mu_matrix,respnm,N_m,n_features,n_m,n_classes);
            #print(var_matrix)
            w_m = gen_weight_array(N_m,n_datapoints);
            #print(w_m)
            logl1 = log_likelihood((mu_matrix,var_matrix,w_m),data_ci,n_m,n_datapoints);
            error = np.abs(logl2-logl1);
            respnm = gen_respnm_expstep(data_ci,mu_matrix,var_matrix,w_m,n_features,n_datapoints,n_features,n_m);
            N_m = gen_N_m(respnm,n_datapoints,n_m)
            logl.append(logl1)
            logl2 = logl1;
        all_ci_mu_matrix[i]=mu_matrix;
        all_ci_var_matrix[i]=var_matrix;
        all_ci_weight_matrix[i]=w_m;
        plt.figure()
        plt.plot(logl)
    logl = np.array(logl);


    plt.show()
    return (all_ci_mu_matrix,all_ci_var_matrix,all_ci_weight_matrix)














def calc_error(assigned_labels,orig_labels):
    correct = np.sum(assigned_labels == orig_labels);
    total = np.size(assigned_labels);
    accuracy = correct / total;
    return accuracy*100;



if __name__ == "__main__":
    [traindata,testdata] = pi.get_data();
    n_classes = 3;
    n_features = 4;
    n_datapoints = 40;
    testdata_labels = testdata[:,4];
    testdata = testdata;
    all_class_labels = np.arange(3);
    n_m = 2;
    n_testpoints = 30;

    priors = inference_prior(traindata);

    class_params = inference_GMM(traindata,all_class_labels,n_classes,n_features,n_datapoints);
    (all_ci_mu_matrix,all_ci_var_matrix,all_ci_weight_matrix)=class_params;
    class_params=(all_class_labels,all_ci_mu_matrix,all_ci_var_matrix);


    assigned_labels = decision_posterior(priors,class_params,testdata[:,0:4],n_classes,30,all_ci_weight_matrix,n_m);
#def decision_posterior(priors,Ci_params,datapoints,n_classes,n_datapoints,w,n_m):

    print(calc_error(assigned_labels,testdata_labels))
