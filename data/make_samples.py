# code to make test/train samples
from data import chris_data as data_maker

def get_sets(params=None,tot_dataset_size=int(15000),ndata=16,usepars=[0,1,2],sigma=0.2,seed=42,r=4):
    if params:
        tot_dataset_size=params['tot_dataset_size']
        ndata=params['ndata']
        usepars=params['usepars']
        sigma=params['sigma']
        seed=params['seed']
        r=params['r']

    # get training set data
    pos_train, labels_train, x, sig_train, parnames = data_maker.generate(
                tot_dataset_size=tot_dataset_size,
                ndata=ndata,
                usepars=usepars,
                sigma=sigma,
                seed=seed
            )
    print('generated training data')

    # get test set data
    pos_test, labels_test, x, sig_test, parnames = data_maker.generate(
                tot_dataset_size=r*r,
                ndata=ndata,
                usepars=usepars,
                sigma=sigma,
                seed=seed
            )
    print('generated testing data')

    return pos_train,sig_train,labels_train,labels_test,pos_test
