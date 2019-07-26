import numpy as np
from time import time
from scipy.stats import gaussian_kde
import h5py

nsg = 5  # max number of sine-gaussian parameters
sg_default = 0.2  # default value of fixed sine-gaussian parameters
parnames = ['A','t0','tau','phi','w']    # parameter names

def overlap(x,y,cnt=0,next_cnt=False):
    """
    compute the overlap between samples from 2 differnt distributions
    """
    if x.shape[1]==1:
        X = np.mgrid[np.min([x[:,0],y[:,0]]):np.max([x[:,0],y[:,0]]):100j]
        positions = np.vstack([X.ravel()])
    #elif nxt_cnt!=True:
    #    X, Y = np.mgrid[np.min([x[:,0],y[:,0]]):np.max([x[:,0],y[:,0]]):100j, np.min([x[:,1],y[:,1]]):np.max([x[:,1],y[:,1]]):100j]
    #    positions = np.vstack([X.ravel(), Y.ravel()])
        #x = np.vstack((x[:,cnt],x[:,nxt_cnt])).T
        #y = np.vstack((y[:,cnt],y[:,nxt_cnt])).T
    elif x.shape[1]==3:
        X, Y, Z = np.mgrid[np.min([x[:,0],y[:,0]]):np.max([x[:,0],y[:,0]]):20j, np.min([x[:,1],y[:,1]]):np.max([x[:,1],y[:,1]]):20j, np.min([x[:,2],y[:,2]]):np.max([x[:,2],y[:,2]]):20j]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    elif x.shape[1]==4:
        #return 0
        X, Y, Z, H = np.mgrid[np.min([x[:,0],y[:,0]]):np.max([x[:,0],y[:,0]]):20j,np.min([x[:,1],y[:,1]]):np.max([x[:,1],y[:,1]]):20j, np.min([x[:,2],y[:,2]]):np.max([x[:,2],y[:,2]]):20j,np.min([x[:,3],y[:,3]]):np.max([x[:,3],y[:,3]]):20j]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), H.ravel()])
    elif x.shape[1]==5:
        X, Y, Z, H, J = np.mgrid[0:1:20j, 0:1:20j, 0:1:20j, 0:1:20j, 0:1:20j]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), H.ravel(), J.ravel()])

    kernel_x = gaussian_kde(x.T)
    Z_x = np.reshape(kernel_x(positions).T, X.shape)
    kernel_y = gaussian_kde(y.T)
    Z_y = np.reshape(kernel_y(positions).T, X.shape)
    n_x = 1.0/np.sum(Z_x)
    n_y = 1.0/np.sum(Z_y)
    print('Computed 4D overlap ...')

    return (np.sum(Z_x*Z_y) / np.sqrt( np.sum(Z_x**2) * np.sum(Z_y**2) ))
    #return (n_y/n_x)*np.sum(Z_x*Z_y)/np.sum(Z_x*Z_x)

def load_training_set(params,train_files,normscales):

    # load generated samples back in
    dataLocations = ["%s" % params['train_set_dir']]
    data = {'x_data_train_h': [], 'y_data_train_lh': []}

    print('Chose file %s' % str(np.random.choice(train_files)))
    data_temp={'x_data_train_h': h5py.File(dataLocations[0]+'/'+np.random.choice(train_files), 'r')['x_data_train_h'][:],
               'y_data_train_lh': h5py.File(dataLocations[0]+'/'+np.random.choice(train_files), 'r')['y_data_train_lh'][:]}
    data['x_data_train_h'] = data_temp['x_data_train_h']
    data['y_data_train_lh'] = data_temp['y_data_train_lh']

    data['x_data_train_h'] = np.array(data['x_data_train_h'])
    data['y_data_train_lh'] = np.array(data['y_data_train_lh'])

    if params['do_normscale']:

        data['x_data_train_h'][:,0]=data['x_data_train_h'][:,0]/normscales[0]
        #data['x_data_train_h'][:,1]=data['x_data_train_h'][:,1]/normscales[1]
        data['x_data_train_h'][:,2]=data['x_data_train_h'][:,2]/normscales[1]
        data['x_data_train_h'][:,3]=data['x_data_train_h'][:,3]/normscales[2]
        data['x_data_train_h'][:,4]=data['x_data_train_h'][:,4]/normscales[3]
    #    data['x_data_train_h'][:,5]=data['x_data_train_h'][:,5]/normscales[5]

    x_data_train_h = data['x_data_train_h']
    y_data_train_lh = data['y_data_train_lh']
    # Remove phase parameter
    x_data_train_h = x_data_train_h[:,[0,2,3,4]]
    x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh

    return x_data_train, y_data_train_l, y_data_train_h, x_data_train_h, y_data_train_lh

