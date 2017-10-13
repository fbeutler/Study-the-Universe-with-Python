import sys, os 
import numpy as np

import emcee
import corner
import scipy.optimize as op
import scipy.interpolate as interp
from scipy.constants import speed_of_light
from numpy.linalg import inv
import itertools

import nbodykit.lab
import nbodykit
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

base_dir = '/path/to/data/'
output_path = base_dir
zmin = 0.5
zmax = 0.75
zmean = 0.61 # see calculation in read_data()
kmin = 0.01
kmax = 0.3

# Download the data and corresponding random BOSS dataset (Two files for North and South)
# wget -N https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz -P path/to/folder/
# wget -N https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz -P path/to/folder/
# wget -N https://data.sdss.org/sas/dr12/boss/lss/random0_DR12v5_CMASSLOWZTOT_North.fits.gz -P path/to/folder/
# wget -N https://data.sdss.org/sas/dr12/boss/lss/random0_DR12v5_CMASSLOWZTOT_South.fits.gz -P path/to/folder/

# Download the simulated BOSS datasets
# wget -N https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-DR12SGC-COMPSAM_V6C.tar.gz -P path/to/folder/
# wget -N https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-DR12NGC-COMPSAM_V6C.tar.gz -P path/to/folder/
# wget -N https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-Randoms-DR12SGC-COMPSAM_V6C_x50.tar.gz -P path/to/folder/
# wget -N https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/Patchy-Mocks-Randoms-DR12NGC-COMPSAM_V6C_x50.tar.gz -P path/to/folder/

def read_sim_data(filename):
    ''' Read the random simulation catalogues from CSV '''
    sim_cat = nbodykit.lab.CSVCatalog(os.path.join(base_dir, filename),\
        names=['RA', 'DEC', 'Z', 'dummy1', 'NZ', 'dummy2', 'veto', 'Weight'])
    print('sim_cat.columns = ', sim_cat.columns)
    sim_cat = sim_cat[(sim_cat['Z'] > zmin) & (sim_cat['Z'] < zmax)]
    sim_cat = sim_cat[(sim_cat['veto'] > 0)]
    sim_cat['WEIGHT_FKP'] = 1./(1. + 10000.*sim_cat['NZ']);
    return sim_cat

def read_sim_ran(filename):
    ''' Read the simulation catalogues from CSV '''
    ran_cat = nbodykit.lab.CSVCatalog(os.path.join(base_dir, filename),\
        names=['RA', 'DEC', 'Z', 'NZ', 'dummy1', 'veto', 'Weight'])
    print('ran_cat.columns = ', ran_cat.columns)
    ran_cat = ran_cat[(ran_cat['Z'] > zmin) & (ran_cat['Z'] < zmax)]
    ran_cat = ran_cat[(ran_cat['veto'] > 0)]
    ran_cat['WEIGHT_FKP'] = 1./(1. + 10000.*ran_cat['NZ']);
    return ran_cat

def read_data(filename):
    ''' Read the data catalogues from FITS '''
    galaxies_cat = nbodykit.lab.FITSCatalog(os.path.join(base_dir, filename))
    print('galaxies_cat.columns = ', galaxies_cat.columns)
    galaxies_cat = galaxies_cat[(galaxies_cat['Z'] > zmin) & (galaxies_cat['Z'] < zmax)]
    galaxies_cat['Weight'] = galaxies_cat['WEIGHT_SYSTOT'] * (galaxies_cat['WEIGHT_NOZ'] + galaxies_cat['WEIGHT_CP'] - 1.0)
    #zmean = np.average(galaxies_cat['Z'].compute(), weights=(galaxies_cat['Weight']*galaxies_cat['WEIGHT_FKP']).compute())
    return galaxies_cat

def read_ran(filename):
    ''' Read the random catalogues from FITS '''
    randoms_cat = nbodykit.lab.FITSCatalog(os.path.join(base_dir, filename))
    print('randoms_cat.columns = ', randoms_cat.columns)
    randoms_cat = randoms_cat[(randoms_cat['Z'] > zmin) & (randoms_cat['Z'] < zmax)]
    return randoms_cat

def get_smooth_model(parameters, x, templates):
    ''' Combine a noBAO model with polynomials and linear bias '''
    polynomials = parameters[1]/x**3 + parameters[2]/x**2 + parameters[3]/x + parameters[4] + parameters[5]*x
    return parameters[0]*parameters[0]*templates['noBAO'](x) + polynomials

def get_shifted_model(parameters, x, templates):
    ''' Calculate a model including a shift given by alpha '''
    model = get_smooth_model(parameters, x, templates)
    return model*(1. + (templates['os_model'](x/parameters[6]) - 1.)*np.exp(-0.5*x**2*parameters[7]**2))

def within_priors(parameters):
    ''' Test for priors '''
    if len(parameters) > 6. and abs(parameters[6] - 1.) > 0.2:
        return False
    elif len(parameters) > 6. and (parameters[7] < 0. or parameters[7] > 20.):
        return False
    else:
        return True

def calc_chi2(parameters, data, templates, func):
    ''' Compares the model with the data '''
    if within_priors(parameters):
        chi2 = 0.
        # Loop over all datasets which are fit together
        for dataset in data:
            model = func(parameters, dataset['k'], templates)
            diff = (model - dataset['pk'])
            chi2 += np.dot(diff,np.dot(dataset['cov_inv'],diff))
        return chi2
    else:
        return 100000.

def get_loglike(parameters, data, templates, func):
    return -0.5*calc_chi2(parameters, data, templates, func)

def calc_pk(cosmo, data, random):
    ''' Calculate the power spectrum '''
    # add Cartesian position column
    data['Position'] = nbodykit.transform.SkyToCartesian(data['RA'],\
     data['DEC'], data['Z'], cosmo=cosmo)
    random['Position'] = nbodykit.transform.SkyToCartesian(random['RA'],\
     random['DEC'], random['Z'], cosmo=cosmo)

    # Combine data and random catalogue
    fkp = nbodykit.lab.FKPCatalog(data, random)
    # Assign point distribution to 3D grid
    mesh = fkp.to_mesh(Nmesh=512, nbar='NZ', comp_weight='Weight', fkp_weight='WEIGHT_FKP') # consider window='tsc', interlaced=True
    # Calculate power spectrum (monopole only)
    r = nbodykit.lab.ConvolvedFFTPower(mesh, poles=[0], dk=0.01, kmin=kmin)
    return r.poles

def process_data(cosmo, tag):
    # Read the BOSS data
    filename = 'galaxy_DR12v5_CMASSLOWZTOT_%s.fits' % tag
    galaxies = read_data(filename)

    # Read the random catalogues for calibration
    filename = 'random0_DR12v5_CMASSLOWZTOT_%s.fits' % tag
    randoms = read_ran(filename)
    return calc_pk(cosmo, galaxies, randoms)

def cmp_pk_with_error(pk1, pk2):
    ''' Compare two power spectra including uncertainties'''
    plt.clf()
    plt.errorbar(pk1['k'], pk1['k']*pk1['pk'], yerr=pk1['k']*np.sqrt(np.diagonal(pk1['cov'])),\
        marker='.', linestyle = 'None', label=pk1['label'])
    plt.errorbar(pk2['k'], pk2['k']*pk2['pk'], yerr=pk2['k']*np.sqrt(np.diagonal(pk2['cov'])),\
        marker='.', linestyle = 'None', label=pk2['label'])
    plt.legend(loc=0)
    plt.xlabel("k [$h$Mpc$^{-1}$]")
    plt.ylabel("kP [$h^{-2}$ Mpc$^2$]")
    plt.xlim(kmin, kmax)
    plt.show()
    return

def cmp_ratio_with_error(pk1, pk2, models=[]):
    ''' Compare two power spectra including uncertainties'''
    plt.clf()
    for model in models:
        plt.plot(model['k'], model['pk'])
    plt.errorbar(pk1['k'], pk1['ratio'], yerr=pk1['ratio_err'], marker='.',\
     linestyle = 'None', label=pk1['label'])
    plt.errorbar(pk2['k'], pk2['ratio'], yerr=pk2['ratio_err'], marker='.',\
     linestyle = 'None', label=pk2['label'])
    plt.axhline(y=1., color='black', linestyle='--')
    plt.legend(loc=0)
    plt.xlabel("k [$h$Mpc$^{-1}$]")
    plt.ylabel("$P(k)/P^{noBAO}(k)$")
    plt.xlim(kmin, kmax)
    plt.show()
    return

def calc_cov(tag, N):
    ''' Read simulation power spectra and return covariance matrix '''
    list_of_pks = []
    for i in range(1, N):
        pk_file = output_path + "/sims/pk_%s_%d.pickle" % (tag, i)
        pk = pickle.load( open( pk_file, "rb" ) )
        P = pk['power_0'].real - pk.attrs['shotnoise']
        # Limit the k range
        P = P[(pk['k'] < kmax)]
        list_of_pks.append(P)
    return np.cov(np.vstack(list_of_pks).T)

def process_sims(cosmo, tag, N):
    ''' Calculate the covariance matrix using the simulated BOSS datasets '''
    # Test whether we have already all power spectra and can directly return the cov
    try:
        return calc_cov(tag, N)
    except:
        pass
    # Read the random catalogue which can be paired with all of the simulated catalogues
    filename = 'Patchy-Mocks-Randoms-DR12%s-COMPSAM_V6C_x50.dat' % tag
    random = read_sim_ran(filename)

    for i in range(1, N):
        pk_file = output_path + "/sims/pk_%s_%d.pickle" % (tag, i)
        print('calculating %s... ' % pk_file)
        # Only calculate power spectra which we don't have on disk
        if not os.path.isfile(pk_file):
            filename = 'Patchy-Mocks-DR12%s-COMPSAM_V6C/Patchy-Mocks-DR12%s-COMPSAM_V6C_%0.4d.dat' % (tag, tag, i)
            sim = read_sim_data(filename)
            pk = calc_pk(cosmo, sim, random)
            # store the power spectra 
            pickle.dump( pk, open( pk_file, "wb" ) )
    return calc_cov(tag, N)

def process_sims_with_MPI(cosmo, tag, N):
    ''' Calculate the covariance matrix using the simulated BOSS datasets '''
    # Test whether we have already all power spectra and can directly return the cov
    try:
        return calc_cov(tag, N)
    except:
        pass
    # this splits the communicator into chunks of size of roughly N
    with nbodykit.lab.TaskManager(1, use_all_cpus=True) as tm:
        # Read the random catalogue which can be paired with all of the simulated catalogues
        filename = 'Patchy-Mocks-Randoms-DR12%s-COMPSAM_V6C_x50.dat' % tag
        random = read_sim_ran(filename)
        # loop over each box number in parallel
        for i in tm.iterate(range(1, N)):
            pk_file = output_path + "/sims/pk_%s_%d.pickle" % (tag, i)
            print('calculating %s... ' % pk_file)
            # Only calculate power spectra which we don't have on disk
            if not os.path.isfile(pk_file):
                filename = 'Patchy-Mocks-DR12%s-COMPSAM_V6C/Patchy-Mocks-DR12%s-COMPSAM_V6C_%0.4d.dat' % (tag, tag, i)
                sim = read_sim_data(filename)
                pk = calc_pk(cosmo, sim, random)
                # store the power spectra 
                pickle.dump( pk, open( pk_file, "wb" ) )
    return calc_cov(tag, N)

def get_oscillation(krange, Pk_class, Pk_without_BAO):
    ''' Get an oscillation only power spectrum '''
    cov_inv = np.identity(len(krange))
    start = [1., 0., 0., 0., 0., 0.]
    result = op.minimize(calc_chi2, start, args=( [{ 'k': krange, 'pk': Pk_class(krange), 'cov_inv': cov_inv }],\
     { 'noBAO': Pk_without_BAO }, get_smooth_model))
    yolin = Pk_class(krange)/get_smooth_model(result["x"], krange, { 'noBAO': Pk_without_BAO })
    return interp.interp1d(krange, yolin)

def get_percentiles(chain, labels):
    ''' Calculate constraints and uncertainties from MCMC chain '''
    per = np.percentile(chain, [50., 15.86555, 84.13445, 2.2775, 97.7225], axis=0)
    per = np.array([per[0], per[0]-per[1], per[2]-per[0], per[0]-per[3], per[4]-per[0]])
    for i in range(0, len(per[0])):
        print("%s = %f +%f -%f +%f -%f" % (labels[i], per[0][i], per[1][i], per[2][i], per[3][i], per[4][i]))
    return per

def inspect_chain(list_of_samplers, labels=[]):
    ''' Print chain properties '''
    Nchains = len(list_of_samplers)
    dim = list_of_samplers[0].chain.shape[2]
    if not labels:
        # set default labels
        labels = [('para_%i' % i) for i in range(0,dim)]

    mergedsamples = []
    for jj in range(0, Nchains):
        chain_length = list_of_samplers[jj].chain.shape[1]
        mergedsamples.extend(list_of_samplers[jj].chain[:, int(chain_length/2):, :].reshape((-1, dim)))

    # write out chain
    res = open("%schain.dat" % output_path, "w")
    for row in mergedsamples:
        for el in row:
            res.write("%f " % el)
        res.write("\n")
    res.close()

    print("length of merged chain = ", len(mergedsamples))
    try:
        for jj in range(0, Nchains):
            print("Mean acceptance fraction for chain ", jj,": ", np.mean(list_of_samplers[jj].acceptance_fraction))
    except Exception as e:
        print("WARNING: %s" % str(e))
    try:
        for jj in range(0, Nchains):
            print("Autocorrelation time for chain ", jj,": ", list_of_samplers[jj].get_autocorr_time())
    except Exception as e:
        print("WARNING: %s" % str(e))

    try:
        fig = corner.corner(mergedsamples, quantiles=[0.16, 0.5, 0.84], plot_density=False,\
            show_titles=True, title_fmt=".3f", labels=labels)
        fig.savefig("%scorner.png" % output_path)
    except Exception as e:
        print("WARNING: %s" % str(e))

    fig, axes = plt.subplots(dim, 1, sharex=True, figsize=(8, 9))
    for i in range(0, dim):
        for jj in range(0, Nchains):
            axes[i].plot(list_of_samplers[jj].chain[:, :, i].T, alpha=0.4)
        #axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(labels[i])
    fig.tight_layout(h_pad=0.0)
    fig.savefig("%stime_series.png" % output_path)

    try:
        return get_percentiles(mergedsamples, labels)
    except Exception as e:
        print("WARNING: %s" % str(e))
        return None

def gelman_rubin_convergence(within_chain_var, mean_chain, chain_length):
    ''' Calculate Gelman & Rubin diagnostic
    # 1. Remove the first half of the current chains
    # 2. Calculate the within chain and between chain variances
    # 3. estimate your variance from the within chain and between chain variance
    # 4. Calculate the potential scale reduction parameter '''
    Nchains = within_chain_var.shape[0]
    dim = within_chain_var.shape[1]
    meanall = np.mean(mean_chain, axis=0)
    W = np.mean(within_chain_var, axis=0)
    B = np.arange(dim,dtype=np.float)
    B.fill(0)
    for jj in range(0, Nchains):
        B = B + chain_length*(meanall - mean_chain[jj])**2/(Nchains-1.)
    estvar = (1. - 1./chain_length)*W + B/chain_length
    return np.sqrt(estvar/W)

def prep_gelman_rubin(sampler):
    dim = sampler.chain.shape[2]
    chain_length = sampler.chain.shape[1]
    chainsamples = sampler.chain[:, int(chain_length/2):, :].reshape((-1, dim))
    within_chain_var = np.var(chainsamples, axis=0)
    mean_chain = np.mean(chainsamples, axis=0)
    return within_chain_var, mean_chain

def runMCMC(start, data, templates):
    ''' Perform MCMC '''
    dim = len(start)
    Nchains = 4
    nwalkers = 20
    ichaincheck = 400
    minlength = 2000
    epsilon = 0.04

    labels = ['b', 'A1', 'A2', 'A3', 'A4', 'A5', 'alpha', 'sigmaNL']
    expected_error = [0.1, 1., 1., 1., 1., 1., 0.05, 0.1]
    # Set up the sampler.
    pos=[]
    list_of_samplers=[]
    for jj in range(0, Nchains):
        pos.append([start + (2.*np.random.random_sample((dim,)) - 1.)*expected_error for i in range(nwalkers)])
        list_of_samplers.append(emcee.EnsembleSampler(nwalkers=nwalkers, dim=dim, lnpostfn=get_loglike,\
         args=(data, templates, get_shifted_model)))

    # Start MCMC
    print("Running MCMC... ")
    within_chain_var = np.zeros((Nchains, dim))
    mean_chain = np.zeros((Nchains, dim))
    scalereduction = np.arange(dim, dtype=np.float)
    scalereduction.fill(2.)

    itercounter = 0
    chainstep = minlength
    while any(abs(1. - scalereduction) > epsilon):
        itercounter += chainstep
        for jj in range(0, Nchains):
            for result in list_of_samplers[jj].sample(pos[jj], iterations=chainstep, storechain=True):
                pos[jj] = result[0]
            # we do the convergence test on the second half of the current chain (itercounter/2)
            within_chain_var[jj], mean_chain[jj] = prep_gelman_rubin(list_of_samplers[jj])
        scalereduction = gelman_rubin_convergence(within_chain_var, mean_chain, int(itercounter/2))
        print("scalereduction = ", scalereduction)
        chainstep = ichaincheck
    # Investigate the chain and print out some metrics
    return inspect_chain(list_of_samplers, labels)

def main():
    cosmo = nbodykit.lab.cosmology.Cosmology(Omega0_cdm=0.26185743, Omega0_b=0.04814257, h=0.676)

    # Get the data power spectra
    pk_file = output_path + "/pk_South.pickle"
    if not os.path.isfile(pk_file):
        pk_south = process_data(cosmo, 'South')
        pickle.dump( pk_south, open( pk_file, "wb" ) )
    else:
        pk_south = pickle.load( open( pk_file, "rb" ) )
    pk_file = output_path + "/pk_North.pickle"
    if not os.path.isfile(pk_file):
        pk_north = process_data(cosmo, 'North')
        pickle.dump( pk_north, open( pk_file, "wb" ) )
    else:
        pk_north = pickle.load( open( pk_file, "rb" ) )
    P_south = pk_south['power_0'].real - pk_south.attrs['shotnoise']
    P_north = pk_north['power_0'].real - pk_north.attrs['shotnoise']

    # Get the covariance metrices
    cov_south = process_sims(cosmo, 'SGC', 1000)
    cov_south_inv = inv(cov_south)
    cov_north = process_sims(cosmo, 'NGC', 1000)
    cov_north_inv = inv(cov_north)

    # Store power spectrum details in a dictionary
    boss_data = [{ 'k': pk_south['k'][(pk_south['k'] < kmax)], 'pk': P_south[(pk_south['k'] < kmax)],\
     'label': 'BOSS south', 'cov': cov_south, 'cov_inv': cov_south_inv},\
     { 'k': pk_north['k'][(pk_north['k'] < kmax)], 'pk': P_north[(pk_north['k'] < kmax)],\
     'label': 'BOSS north', 'cov': cov_north, 'cov_inv': cov_north_inv }]
    #cmp_pk_with_error(pk_plot1, pk_plot2)

    # Isolate the BAO signal 
    Pk_without_BAO = nbodykit.lab.cosmology.power.linear.LinearPower(cosmo, redshift=0, transfer='NoWiggleEisensteinHu')
    Pk_class = nbodykit.lab.cosmology.power.linear.LinearPower(cosmo, redshift=0, transfer='CLASS')
    krange = np.arange(0.001, 0.5, 0.001)
    os_model = get_oscillation(krange, Pk_class, Pk_without_BAO)
    start = [2.37, -0.076, 38., -3547., 15760., -22622., 1., 9.41]
    result = op.minimize(calc_chi2, start, args=(boss_data, { 'noBAO': Pk_without_BAO, 'os_model': os_model },\
     get_shifted_model))
    print("result['x'] = ", result['x'])

    krange = np.arange(kmin, kmax, 0.001)
    best_fit_model = get_shifted_model(result["x"], krange, { 'noBAO': Pk_without_BAO, 'os_model': os_model })/get_smooth_model(result["x"], krange, { 'noBAO': Pk_without_BAO })
    boss_data[0]['ratio'] = boss_data[0]['pk']/get_smooth_model(result["x"], boss_data[0]['k'], { 'noBAO': Pk_without_BAO })
    boss_data[0]['ratio_err'] = np.sqrt(np.diagonal(boss_data[0]['cov']))/get_smooth_model(result["x"], boss_data[0]['k'], { 'noBAO': Pk_without_BAO })
    boss_data[1]['ratio'] = boss_data[1]['pk']/get_smooth_model(result["x"], boss_data[1]['k'], { 'noBAO': Pk_without_BAO })
    boss_data[1]['ratio_err'] = np.sqrt(np.diagonal(boss_data[1]['cov']))/get_smooth_model(result["x"], boss_data[1]['k'], { 'noBAO': Pk_without_BAO })
    cmp_ratio_with_error(boss_data[0], boss_data[1], [{ 'k': krange, 'pk': best_fit_model }])

    per = runMCMC(result["x"], boss_data, { 'noBAO': Pk_without_BAO, 'os_model': os_model })
    if per is not None:
        Hz = cosmo.efunc(zmean)*cosmo.H0*cosmo.h
        DC = (1.+zmean)*cosmo.angular_diameter_distance(zmean)/cosmo.h
        c_km_s = (speed_of_light/1000.)
        DVfid = ( DC**2*(zmean*c_km_s/Hz) )**(1./3.)
        DV = per[:,6]*DVfid
        print("DV = %f +%f -%f +%f -%f\n" % (DV[0], DV[1], DV[2], DV[3], DV[4]))
    else:
        print("Problem with MCMC chain")
    return 

if __name__ == '__main__':
    main()
