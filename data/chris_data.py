import numpy as np
from time import time
from scipy.stats import gaussian_kde

nsg = 5  # max number of sine-gaussian parameters
sg_default = 0.2  # default value of fixed sine-gaussian parameters
parnames = ['A','t0','tau','phi','w']    # parameter names

def sg(x,pars):
    """
    generates a noise-free sine-gaussian signal
    """
    A,t0,tau,p,w = pars
    fnyq = 0.5*len(x)
    return A*np.sin(2.0*np.pi*(w*fnyq*x + p))*np.exp(-((x-t0)/tau)**2)

def generate(tot_dataset_size,ndata=8,usepars=[0,1],sigma=0.1,seed=0):

    np.random.seed(seed)
    N = tot_dataset_size
    ndim = len(usepars)

    # fill in parameters
    bigpars = sg_default*np.ones((N,nsg))
    pars = np.random.uniform(0,1,size=(N,ndim))
    bigpars[:,usepars] = pars
    names = [parnames[int(i)] for i in usepars]

    # make y = sine-gaussian  + noise
    noise = np.random.normal(loc=0.0,scale=sigma,size=(N,ndata))
    xvec = np.arange(ndata)/float(ndata)
    sig = np.array([sg(xvec,p) for p in bigpars])
    data = sig + noise

    # randomise the data 
    shuffling = np.random.permutation(N)
    pars = pars[shuffling]
    data = data[shuffling]
    sig = sig[shuffling]

    return pars, data, xvec, sig, names

def mcmc_sampler(r,N_samp,ndim_x,labels_test,sigma,usepars,n_burnin=2000):
    """
    Iterate over all test samples to produce mcmc samples
    """
    
    def get_lik(ydata,sigma=0.2,usepars=[0,1],Nsamp=1000):
        """
        returns samples from the posterior obtained using trusted 
        techniques
        """

        def logposterior(theta, data, sigma, x, usepars):
            """
            The natural logarithm of the joint posterior.
            Args:
                theta (tuple): a sample containing individual parameter values
                data (list): the set of data/observations
                sigma (float): the standard deviation of the data points
                x (list): the abscissa values at which the data/model is defined
            """

            lp = logprior(theta) # get the prior

            # if the prior is not finite return a probability of zero (log probability of -inf)
            if not np.isfinite(lp):
                return -np.inf

            # return the likeihood times the prior (log likelihood plus the log prior)
            return lp + loglikelihood(theta, data, sigma, x, usepars)

        def loglikelihood(theta, data, sigma, x, usepars):
            """
            The natural logarithm of the joint likelihood.
            Args:
                theta (tuple): a sample containing individual parameter values
                data (list): the set of data/observations
                sigma (float): the standard deviation of the data points
                x (list): the abscissa values at which the data/model is defined
            Note:
                We do not include the normalisation constants (as discussed above).
            """

            # fill in the parameters and evaluate the model
            pars = sg_default*np.ones(nsg)
            pars[usepars] = theta
            md = sg(x,pars)

            # return the log likelihood
            return -0.5*np.sum(((md - data)/sigma)**2)

        def logprior(theta):
            """
            The natural logarithm of the prior probability.
            Args:
                theta (tuple): a sample containing individual parameter values
            Note:
                We can ignore the normalisations of the prior here.
            """

            if np.any(theta<0) or np.any(theta>1.0):
                return -np.inf
            return 0.0

        ndims = len(usepars)        # number of search dimensions
        N = ydata.size              # length of timeseries data
        x = np.arange(N)/float(N)   # time vector
        Nens = 100                  # number of ensemble points
        Nburnin = n_burnin               # number of burn-in samples
        Nsamples = 500              # number of final posterior samples
        p0 = [np.random.rand(ndims) for i in range(Nens)]

        import emcee # import the emcee package
        print('emcee version: {}'.format(emcee.__version__))

        # set additional args for the posterior (the data, the noise std. dev., and the abscissa)
        argslist = (ydata, sigma, x, usepars)

        # set up the sampler
        sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)

        # pass the initial samples and total number of samples required
        t0 = time() # start time
        sampler.run_mcmc(p0, Nsamples+Nburnin);
        t1 = time()

        timeemcee = (t1-t0)
        print("Time taken to run 'emcee' is {} seconds".format(timeemcee))

        # extract the samples (removing the burn-in)
        samples_emcee = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))
        idx = np.random.randint(low=0,high=samples_emcee.shape[0],size=Nsamp)
        return samples_emcee[idx,:]

    cnt = 0
    samples = np.zeros((r*r,N_samp,ndim_x))
    for i in range(r):
        for j in range(r):
            samples[cnt,:,:] = get_lik(np.array(labels_test[cnt,:]).flatten(),sigma=sigma,usepars=usepars,Nsamp=N_samp)
            cnt += 1
    return samples

def overlap(x,y):
    """
    compute the overlap between samples from 2 differnt distributions
    """
    if x.shape[1]==1:
        X = np.mgrid[0:1:100j] 
        positions = np.vstack([X.ravel()])

    elif x.shape[1]==2:
        X, Y = np.mgrid[0:1:20j, 0:1:20j]
        positions = np.vstack([X.ravel(), Y.ravel()])
    elif x.shape[1]==3:
        X, Y, Z = np.mgrid[0:1:20j, 0:1:20j, 0:1:20j]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    elif x.shape[1]==4:
        X, Y, Z, H = np.mgrid[0:1:20j, 0:1:20j, 0:1:20j, 0:1:20j]
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
    print('Computed overlap ...')

    return (np.sum(Z_x*Z_y) / np.sqrt( np.sum(Z_x**2) * np.sum(Z_y**2) ))
    #return (n_y/n_x)*np.sum(Z_x*Z_y)/np.sum(Z_x*Z_x)

#def get_lik(ydata,n_grid=64,sig_model='sg',sigma=None,xvec=None,bound=[0,1,0,1,0,1]):
#
#    mcx = np.linspace(bound[0],bound[1],n_grid)              # vector of mu values
#    mcy = np.linspace(bound[2],bound[3],n_grid)
#    dmcx = mcx[1]-mcx[0]                       # mu spacing
#    dmcy = mcy[1]-mcy[0]
#    mv, cv = np.meshgrid(mcx,mcy)        # combine into meshed variables
#
#    res = np.zeros((n_grid,n_grid))
#    if sig_model=='slope':
#        for i,c in enumerate(mcy):
#            res[i,:] = np.array([np.sum(((ydata-m*xvec-c)/sigma)**2) for m in mcx])
#        res = np.exp(-0.5*res)
#    elif sig_model=='sg':
#        w = 6.0*np.pi
#        p = 1.0
#        tau = 0.25
#        for i,t in enumerate(mcy):
#            res[i,:] = np.array([np.sum(((ydata - A*np.sin(w*xvec + p)*np.exp(-((xvec-t)/tau)**2))/sigma)**2) for A in mcx])
#        res = np.exp(-0.5*res)
#
#    # normalise the posterior
#    res /= (np.sum(res.flatten())*dmcx*dmcy)
#
#    # compute integrated probability outwards from max point
#    res = res.flatten()
#    idx = np.argsort(res)[::-1]
#    prob = np.zeros(n_grid*n_grid)
#    prob[idx] = np.cumsum(res[idx])*dmcx*dmcy
#    prob = prob.reshape(n_grid,n_grid)
#    res = res.reshape(n_grid,n_grid)
#    return mcx, mcy, prob
