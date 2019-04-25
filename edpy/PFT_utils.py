## A module for understand and parameterize PFTs in ED 
## Currently, it can
## (1) read key PFT parameters from xml
## (2) create trees of different sizes
## (3) calculate key rates (mainly growth) for trees of different sizes
#  Author: Xiangtao Xu

import numpy as np
import pandas as pd
import sys
from lxml import etree
import h5py

# some constants
# some constants to calculate vpd
c0 = .6105851e+03
c1 = .4440316e+02
c2 = .1430341e+01
c3 = .2641412e-01
c4 = .2995057e-03
c5 = .2031998e-05
c6 = .6936113e-08
c7 = .2564861e-11
c8 =-.3704404e-13

month_strs = np.array(
['JAN', 'FEB', 'MAR', 'APR'
,'MAY', 'JUN', 'JUL', 'AUG'
,'SEP', 'OCT', 'NOV', 'DEC']
)


# PFT class

class PFT(object):
    '''
        Store the functional traits of each PFT
    '''

    #####################
    # Any cross-PFT constants
    #####################
    C2B = 2.  # Carbon to dry biomass ratio
    UMOL2KGC   = 1e-9 * 12 
    ref_tmp=15.
    Q10=2.
    low_tmp=8.
    high_tmp=45.
    decay_e=0.4

    # some constants
    aO2 = 210.
    aCO2 = 390.
    R       = 8.3144598     # J/K/mol, universal gas constant
    aPres   = 1.01325e5     # Pa, atmospheric pressure

    phiPSII    = .7         # Maximum quantum yield of PSII
    thetaPSII  = .7         # curvature factor for light response
    g1      = 3.77          # parameter for Medlyn stomatal scheme, based on Lin et al. 2015

    alphaPar = 1. - 0.15    
    # leaf absorptance of PAR, this come from de Pury and Farquhar 1997 paper

    betaPar  = 0.5          
    # fraction of PAR that reaches PSII, from Bernacchi et al. 2013

    Jmax25B = np.array([0.,1.67])  
    # coefficients of linear relationship between Jmax25 and Vcmax25
    # Refs: Kattge and Knorr 2007
    deltaHKc        = 79.43 * 1e3
    deltaHKo        = 36.38 * 1e3
    deltaHGammaStar = 37.83 * 1e3

    gammaStarRef    =   42.75       # umol/mol
    KcRef           =   404.9       # umol/mol
    KoRef           =   278.4       # mmol/mol

    # following parameter is from Bernacchi et al. 2003 Table 2
    thetaPSIIb      = np.array([0.76,0.018,-3.7e-4])      # Growth Temperature = 25 degC
    phiPSIIb        = np.array([0.352,0.022,-3.42e-4])




    # construction function
    # define all the parameters to include
    def __init__(self,
                 xml_name : 'name of input xml file',
                 pft_num : 'pft number in the xml file'):

        # first read the xml
        params_root = etree.parse(xml_name).getroot()
        # loop over all the pfts to find the right PFT to use
        pft_list = params_root.findall('pft')
        find_pft = False
        for ipft, pft_element  in enumerate(pft_list):
            if int(pft_element.find('num').text) == pft_num:
                find_pft = True
                break

        if find_pft is False:
            # not find the pft print an error
            sys.exit("Can't find PFT {:d} in {:s} ".format(
                        pft_num,xml_name))



        ##########################
        # After here, we have find the right structure to use
        # It is stored in pft_element


        # we need to read physiology setups as well
#        physiology_element = params_root.find('physiology')
#        self.istomata_scheme = int(physiology_element.find('istomata_scheme').text)
#        self.istruct_growth_scheme = int(physiology_element.find('istruct_growth_scheme').text)
#        self.trait_plasticity_scheme = int(physiology_element.find('trait_plasticity_scheme').text)

        # we define the parameters from the pft_element
        self.num = pft_num
        self.is_tropical = int(pft_element.find('is_tropical').text)
        self.is_grass = int(pft_element.find('is_grass').text)
        self.is_liana = int(pft_element.find('is_liana').text)
        
        # Photosynthetic
        self.Vm0 = float(pft_element.find('Vm0').text)
        self.Vm_low_temp = float(pft_element.find('Vm_low_temp').text)
        self.Vm_high_temp = float(pft_element.find('Vm_high_temp').text)
        self.Vm_q10 = float(pft_element.find('vm_q10').text)
        self.quantum_efficiency = float(pft_element.find('quantum_efficiency').text)

        # Respiration
        self.Rd0 = float(pft_element.find('Rd0').text)
        self.Rd_q10 = float(pft_element.find('Rd_q10').text)
        self.growth_resp_factor = float(pft_element.find('growth_resp_factor').text)
        self.root_respiration_factor = float(pft_element.find('root_respiration_factor').text)
        self.stem_respiration_factor = float(pft_element.find('stem_respiration_factor').text)
        self.rrf_q10 = float(pft_element.find('rrf_q10').text)

        # Stomatal
        self.D0 = float(pft_element.find('D0').text)
        self.stomatal_slope = float(pft_element.find('stomatal_slope').text)

        # Turnover
        self.leaf_turnover_rate = float(pft_element.find('leaf_turnover_rate').text)
        self.root_turnover_rate = float(pft_element.find('root_turnover_rate').text)
        self.storage_turnover_rate = float(pft_element.find('storage_turnover_rate').text)

        # Structural
        self.SLA = float(pft_element.find('SLA').text)
        self.rho = float(pft_element.find('rho').text)
        self.q = float(pft_element.find('q').text)
        self.qsw = float(pft_element.find('qsw').text)

        # Allometry
        self.b1Ht = float(pft_element.find('b1Ht').text)
        self.b2Ht = float(pft_element.find('b2Ht').text)
        self.hgt_ref = float(pft_element.find('hgt_ref').text)
        self.hgt_min = float(pft_element.find('hgt_min').text)
        self.hgt_max = float(pft_element.find('hgt_max').text)
        self.min_dbh = float(pft_element.find('min_dbh').text)
        self.dbh_crit = float(pft_element.find('dbh_crit').text)
        self.b1Bl_small = float(pft_element.find('b1Bl_small').text)
        self.b1Bl_large = float(pft_element.find('b1Bl_large').text)
        self.b2Bl_small = float(pft_element.find('b2Bl_small').text)
        self.b2Bl_large = float(pft_element.find('b2Bl_large').text)
        self.b2Bl_hite = float(pft_element.find('b2Bl_hite').text)
        self.b1Bs_small = float(pft_element.find('b1Bs_small').text)
        self.b2Bs_small = float(pft_element.find('b2Bs_small').text)
        self.b1Bs_large = float(pft_element.find('b1Bs_large').text)
        self.b2Bs_large = float(pft_element.find('b2Bs_large').text)
        self.b2Bs_hite = float(pft_element.find('b2Bs_hite').text)
        self.b1SA = float(pft_element.find('b1SA').text)
        self.b2SA = float(pft_element.find('b2SA').text)
        #self.b1CA = float(pft_element.find('b1CA').text)
        self.agf_bs = float(pft_element.find('agf_bs').text)
        self.repro_min_h = float(pft_element.find('repro_min_h').text)
        self.r_fract = float(pft_element.find('r_fract').text)


        # some variables for experiments
        self.repro_method = 'ED2.1'  # use the abrupt change in ED2.1
        self.repro_hmax = 35.

        # finished
        return



    ## methods

    ## Allometric equations
    # Here we only use the new scheme (use both hite and dbh)
    def h2dbh(self,h):
        if type(h).__module__ != np.__name__:
            h = np.array(h)
        return np.exp((np.log(h)-self.b1Ht) / self.b2Ht)
    def dbh2h(self,dbh):
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        return np.exp(self.b1Ht + self.b2Ht * np.log(
                        np.minimum(dbh,self.dbh_crit) 
                        ))
    def size2bstem(self,dbh,h):
        if type(h).__module__ != np.__name__:
            h = np.array(h)
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        bstem = (self.b1Bs_small / PFT.C2B * 
                 (dbh ** self.b2Bs_small) *
                 (h ** self.b2Bs_hite)
                )
        dbh_mask = (dbh > self.dbh_crit)
        if np.sum(dbh_mask) > 0:
            bstem[dbh_mask] = (self.b1Bs_large / PFT.C2B * 
                               (dbh[dbh_mask] ** self.b2Bs_large) *
                               (h[dbh_mask] ** self.b2Bs_hite)
                              )
        return bstem

    def bstem2dbh(self,bstem):
        if type(bstem).__module__ != np.__name__:
            bstem = np.array(bstem)

        # calculate bs_crit
        bs_crit = self.size2bstem([self.dbh_crit],[self.hgt_max])

        # default small trees
        dbh = np.exp( ( np.log(PFT.C2B * bstem)
                      - np.log(self.b1Bs_small) - self.b2Bs_hite * self.b1Ht)
                    / (self.b2Bs_small + self.b2Bs_hite * self.b2Ht)
                    )
        # deal with
        bs_mask = (bstem >= bs_crit)
        if np.sum(bs_mask) > 0:
            dbh[bs_mask] = np.exp(
                ( np.log(PFT.C2B * bstem[bs_mask])
                - np.log(self.b1Bs_large) - self.b2Bs_hite * np.log(self.hgt_max))
                / self.b2Bs_large)

        return dbh

    def dbh2sf(self,dbh):
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        if self.is_grass == 1:
            return np.ones_like(dbh)

        return np.minimum(1.,
                         (self.b1SA * dbh ** self.b2SA) / 
                         (np.pi / 4. * dbh ** 2))

    def size2bleaf(self,dbh,h):
        if type(h).__module__ != np.__name__:
            h = np.array(h)
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        return (self.b1Bl_large / PFT.C2B / self.SLA * 
                (np.minimum(self.dbh_crit,dbh) ** self.b2Bl_large) *
                (h ** self.b2Bl_hite))

    def size2broot(self,dbh,h):
        if type(h).__module__ != np.__name__:
            h = np.array(h)
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        return self.size2bleaf(dbh,h) * self.q

    def size2bstorage(self,dbh,h):
        if type(h).__module__ != np.__name__:
            h = np.array(h)
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        return self.size2bleaf(dbh,h) * (1. + self.q)

    def size2btotal(self,dbh,h):
        if type(h).__module__ != np.__name__:
            h = np.array(h)
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        return (self.size2bstem(dbh,h)
               +self.size2bleaf(dbh,h)
               +self.size2broot(dbh,h)
               +self.size2bstorage(dbh,h)
               )

    def size2repro_frac(self,dbh,h):
        if type(h).__module__ != np.__name__:
            h = np.array(h)
        if type(dbh).__module__ != np.__name__:
            dbh = np.array(dbh)

        repro_frac = np.zeros_like(h)
        if self.repro_method == 'ED2.1':
            # old ED2.1 scheme
            repro_mask = (h >= self.repro_min_h)
            repro_frac[repro_mask] = self.r_fract
        elif self.repro_method == 'MM':
            # Michaelis-Menton
            repro_frac = (
                np.maximum(0.,h-self.repro_min_h) / 
                (np.maximum(0.,h-self.repro_min_h) + self.repro_hmax)
                         ) * 0.4
        elif self.repro_method == 'linear':
            # Linear
            repro_frac = np.minimum(1.,
                np.maximum(0.,h-self.repro_min_h) / 
                (self.repro_hmax - self.repro_min_h)
                         ) * 0.4

        return repro_frac



#########
# A class of trees to record size and conduct photosynthesis

class Trees(object):


    @classmethod
    def temperature_dependence(cls,leaf_tmp,func_type='Q10',
                               Q10=PFT.Q10,ref_tmp=PFT.ref_tmp,
                               low_tmp=PFT.low_tmp,high_tmp=PFT.high_tmp,
                               deltaH=PFT.deltaHKc,polyB=PFT.thetaPSIIb):
        if type(leaf_tmp).__module__ != np.__name__:
            leaf_tmp = np.array(leaf_tmp)

        if func_type == 'Q10':
            temp_scale = Q10 ** (0.1 * (leaf_tmp - ref_tmp))
            explow = np.exp(PFT.decay_e * (low_tmp - leaf_tmp))
            exphigh = np.exp(PFT.decay_e * (leaf_tmp - high_tmp))

            temp_scale /= ((1. + explow) * (1. + exphigh))
        elif func_type == 'B':
            # eqn 10 in Bernacchi et al. 2013
            temp_scale = np.exp((leaf_tmp - ref_tmp) 
                          * deltaH 
                          / (PFT.R * (leaf_tmp+273.15) * (ref_tmp+273.15))
                         )
        elif func_type == 'P':
            # polynomial, used for theta and phi PSII, from Bernacchi et al. 2003
            temp_scale = polyB[0] + polyB[1]*leaf_tmp + polyB[2]*leaf_tmp**2.

        return temp_scale

    # a class method for stomatal conductance
    @classmethod
    def get_iCO2(cls,vpd,aCO2=PFT.aCO2):
        # based on Medlyn et al. 2012; Lin et al. 2015 NCC
        return aCO2 * (1 - 1./(1. + PFT.g1/(vpd/1000.) ** 0.5))

    # a class method for photosynthesis
    @classmethod
    def farq_photosynthesis(cls,I,iCO2,
                            phiPSII,thetaPSII,gammaStar,
                            Jmax,Vcmax,Kc,Ko,aO2 = PFT.aO2,pres=PFT.aPres):

        iCO2    = iCO2 * pres / PFT.aPres
        iO2     = aO2 * pres / PFT.aPres

        # model photosynthesis based on Bernacchi et al. 2013
        # 1. Rubisco-limited photosynthesis
        Wc      = Vcmax * iCO2 / (iCO2 + Kc*(1 + iO2/Ko)) \
                  * (1 - gammaStar/iCO2)

        # 2. TPU-limited photosynthesis
        Wp      = 0.5 * Vcmax

        # 3. RuBP-limited phtosynthesis (light)
        Iused   = I * PFT.alphaPar * PFT.betaPar * phiPSII

        Jrate   = (1. / (2. * thetaPSII)) \
                  * (Iused + Jmax 
                     - (
                        (Iused + Jmax)** 2. - 4. * thetaPSII * Iused * Jmax
                       ) ** 0.5
                    )
        Wj      = Jrate * iCO2 / (4.*iCO2 + 8.*gammaStar) * (1 - gammaStar/iCO2)

        At      = np.amin(
                    np.reshape(
                    np.concatenate(
                    (Wc,Wj,Wp))
                    ,(3,Wc.shape[0]))
                    ,axis=0)

        return (At,Wc,Wj,Wp)

        
    ##############################################################################
    # initialization
    def __init__(self,
                 PFT,
                 dbh = np.arange(0.5,200.1,0.5)
                ):
        self.PFT    = PFT
        self.dbh    = dbh
        self.ba     = np.pi / 4. * self.dbh ** 2
        self.h      = self.PFT.dbh2h(self.dbh)
        self.bleaf  = self.PFT.size2bleaf(self.dbh,self.h)
        self.la     = self.bleaf * self.PFT.SLA
        self.bstem  = self.PFT.size2bstem(self.dbh,self.h)
        self.sf     = self.PFT.dbh2sf(self.dbh)
        self.broot  = self.PFT.q * self.bleaf
        self.bstorage = (1. + self.PFT.q) * self.bleaf

        self.tree_num = len(self.dbh)

        # create structure for temperature, vpd, light
        # for growth calculations
        self.leaf_tmp = np.zeros_like(self.dbh)
        self.leaf_vpd = np.zeros_like(self.dbh)
        self.leaf_par = np.zeros_like(self.dbh)

        # create structure for carbon balance
        self.gpp = np.zeros_like(self.dbh)
        self.resp_leaf = np.zeros_like(self.dbh)
        self.resp_root = np.zeros_like(self.dbh)
        self.resp_stem = np.zeros_like(self.dbh)
        self.resp_growth = np.zeros_like(self.dbh)
        self.npp = np.zeros_like(self.dbh)
        self.turnover_leaf = np.zeros_like(self.dbh)
        self.turnover_root = np.zeros_like(self.dbh)
        self.turnover_stem = np.zeros_like(self.dbh)
        self.turnover_bstorage = np.zeros_like(self.dbh)
        self.cb = np.zeros_like(self.dbh)
        self.repro = np.zeros_like(self.dbh)
        self.dbtotal_dt = np.zeros_like(self.dbh) # total growth in biomass

        # create structure for growth and reproduction
        self.dbstem_dt  = np.zeros_like(self.dbh) # growth in stems
        self.dbleaf_dt  = np.zeros_like(self.dbh) # growth in leaf
        self.dbroot_dt  = np.zeros_like(self.dbh) # growth in root
        self.dbstorage_dt  = np.zeros_like(self.dbh) # growth in storage
        self.ddbh_dt = np.zeros_like(self.dbh) # growth in dbh
        self.dba_dt  = np.zeros_like(self.dbh) # growth in basal area
        self.dsa_dt  = np.zeros_like(self.dbh) # growth in sapwoo area
        self.dla_dt  = np.zeros_like(self.dbh) # growth in leaf area
        self.dh_dt   = np.zeros_like(self.dbh) # growth in height
    ##############################################################################

    ##############################################################################
    # instance method, update climatic variables
    def update_climate(self,tmp,par,vpd):
        if len(tmp) == 1:
            self.leaf_tmp[:] = tmp
        elif len(tmp) == self.tree_num:
            self.leaf_tmp = np.array(tmp)

        if len(par) == 1:
            self.leaf_par[:] = par
        elif len(par) == self.tree_num:
            self.leaf_par = np.array(par)
        
        if len(vpd) == 1:
            self.leaf_vpd[:] = vpd
        elif len(vpd) == self.tree_num:
            self.leaf_vpd = np.array(vpd)

    ##############################################################################


    ##############################################################################
    # instance method, get annual carbon balance for each tree
    def get_cb(self,met_dir,met_pf,met_res,years,row=0,col=0,
               shading = 1.0):
        # read the met data from a specific ED2 HEADER file
        year_num = len(years)

        for iyear, year in enumerate(years):
            for imonth, month in enumerate(month_strs):
                met_fn = met_dir + met_pf + '{:d}{:s}.h5'.format(year,month)
                h5in    = h5py.File(met_fn,'r')
                #print(met_fn)
                #print(np.array(h5in['tmp']).shape)
                air_tmp = np.array(h5in['tmp'][row,col,:]).ravel() - 273.15
                #print(air_tmp.shape)
                air_pres = np.array(h5in['pres'][row,col,:]).ravel() 
                air_sh = np.array(h5in['sh'][row,col,:]).ravel() 
                x = air_tmp
                es = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * (c6 + x * (c7 + x * c8)))))))
                e = air_sh * air_pres / (0.622 + 0.378 * air_sh)
                air_vpd = es - e

                vbdsf = np.array(h5in['vbdsf'][row,col,:]).ravel()
                vddsf = np.array(h5in['vddsf'][row,col,:]).ravel()
                air_par = (vbdsf+vddsf) * 4.6 * shading
                h5in.close()

                # for now we assume leaf and air are synced
                leaf_tmps = air_tmp.copy()
                leaf_vpds = air_vpd.copy()
                leaf_pars = air_par.copy()
    
                # we need to conduct photosynthesis
                for itime in np.arange(len(leaf_tmps)):
                    leaf_tmp = np.ones_like(self.dbh) * leaf_tmps[itime]
                    leaf_vpd = np.ones_like(self.dbh) * leaf_vpds[itime]
                    leaf_par = np.ones_like(self.dbh) * leaf_pars[itime]

                    # update all the parameters for photosynthesis
                    gammaStar= (
                        PFT.gammaStarRef
                      * self.temperature_dependence(leaf_tmp,ref_tmp=25.,
                        deltaH=PFT.deltaHGammaStar,func_type='B'))
                    Kc= (
                        PFT.KcRef
                      * self.temperature_dependence(leaf_tmp,ref_tmp=25.,
                        deltaH=PFT.deltaHKc,func_type='B'))
                    Ko= (
                        PFT.KoRef
                      * self.temperature_dependence(leaf_tmp,ref_tmp=25.,
                        deltaH=PFT.deltaHKo,func_type='B'))
                    thetaPSII= (
                        self.temperature_dependence(leaf_tmp,ref_tmp=25.,
                        polyB=PFT.thetaPSIIb,func_type='P'))
                    phiPSII= (
                        self.temperature_dependence(leaf_tmp,ref_tmp=25.,
                        polyB=PFT.phiPSIIb,func_type='P'))
                    Vcmax = (self.PFT.Vm0 * 
                             self.temperature_dependence(
                                 leaf_tmp,func_type='Q10',
                                 Q10=self.PFT.Vm_q10,ref_tmp=15.,
                                 low_tmp=self.PFT.Vm_low_temp,
                                 high_tmp=self.PFT.Vm_high_temp))
                    Rd = (self.PFT.Rd0 * 
                             self.temperature_dependence(
                                 leaf_tmp,func_type='Q10',
                                 Q10=self.PFT.Rd_q10,ref_tmp=15.,
                                 low_tmp=self.PFT.Vm_low_temp,
                                 high_tmp=self.PFT.Vm_high_temp))
                    rrf = (self.PFT.root_respiration_factor * 
                             self.temperature_dependence(
                                 leaf_tmp,func_type='Q10',
                                 Q10=self.PFT.rrf_q10,ref_tmp=15.,
                                 low_tmp=self.PFT.Vm_low_temp,
                                 high_tmp=self.PFT.Vm_high_temp))

                    srf = (self.PFT.stem_respiration_factor * 
                             self.temperature_dependence(
                                 leaf_tmp,func_type='Q10',
                                 Q10=self.PFT.rrf_q10,ref_tmp=15.,
                                 low_tmp=self.PFT.Vm_low_temp,
                                 high_tmp=self.PFT.Vm_high_temp))

                    Vcmax25 = (self.PFT.Vm0 * 
                             self.temperature_dependence(
                                 [25.],func_type='Q10',
                                 Q10=self.PFT.Vm_q10,ref_tmp=15.,
                                 low_tmp=self.PFT.Vm_low_temp,
                                 high_tmp=self.PFT.Vm_high_temp))
                    Jmax25 = PFT.Jmax25B[0] + PFT.Jmax25B[1] * Vcmax25[0]
                    Jmax = (Jmax25 * 
                             self.temperature_dependence(
                                 leaf_tmp,func_type='Q10',
                                 Q10=self.PFT.Vm_q10,ref_tmp=25.,
                                 low_tmp=self.PFT.Vm_low_temp,
                                 high_tmp=self.PFT.Vm_high_temp))



                    # Get iCO2
                    iCO2 = self.get_iCO2(leaf_vpd)
                    # conduct photosynthesis
                    At,Wc,Wj,Wp = self.farq_photosynthesis(
                                  leaf_par,iCO2,phiPSII,thetaPSII,
                                  gammaStar,Jmax,Vcmax,Kc,Ko) 
 
                    # record in the different fluxes variables
                    self.gpp += (At * PFT.UMOL2KGC * met_res * self.la)  # kgC/pl
                    self.resp_leaf += (Rd * PFT.UMOL2KGC * met_res * self.la) # kgC/pl
                    self.resp_root += (self.broot * rrf * met_res * PFT.UMOL2KGC) # kgC/pl
                    self.resp_stem += ((srf + 0.0041 * self.dbh) * self.dbh/100. * np.pi * self.h
                                           * met_res * PFT.UMOL2KGC) # kgC/pl Lavigne et al. 2004
                    # we will calculate growth respiration later...


        # convert to kgC/pl/yr
        self.gpp /= (year_num)
        self.resp_leaf /= (year_num)
        self.resp_root /= (year_num)
        self.resp_stem /= (year_num)
        # calculate resp_growth
#        self.resp_growth = (
#            (self.gpp - self.resp_leaf - self.resp_root - self.resp_stem) * 
#            self.PFT.growth_resp_factor)
        # calculate NPP

        # calculate turnover
        self.turnover_leaf = self.PFT.leaf_turnover_rate * self.bleaf
        self.turnover_root = self.PFT.root_turnover_rate * self.broot   
        self.turnover_bstorage = self.PFT.storage_turnover_rate * self.bstorage
        self.turnover_stem = 0. # for now no turnover rate for stem

        self.resp_growth = np.exp((
            np.maximum(0.,np.minimum(4.,
            (self.gpp - self.resp_leaf - self.resp_root - self.resp_stem
                      - self.turnover_leaf - self.turnover_root - self.turnover_stem
                      - self.turnover_bstorage) / 
            (self.bleaf + self.bstem + self.broot + self.bstorage))) * 0.5
        )) * self.resp_stem

        self.npp = self.gpp - (self.resp_leaf + self.resp_root + self.resp_stem + self.resp_growth)
        # calculate total carbon balance that can be used for reproduction and growth
        self.cb = self.npp - (self.turnover_leaf + self.turnover_root
                             +self.turnover_stem + self.turnover_bstorage)

        # calculate reproduction and dbtotal_dt
        self.repro = self.cb * self.PFT.size2repro_frac(self.dbh,self.h)
        self.dbtotal_dt = self.cb - self.repro 


        return
    
    def get_growth(self):
        # called after get_annual_cb
        # calcualte all the growth rates from the cb rate
        # most importantly get dbstem_dt, dbleaf_dt, dbroot_dt and dbstorage_dt

        btotal = self.bleaf + self.bstem + self.broot + self.bstorage

        # first get the current bstem to btotal ratio
        # we assume all growth go to bstem, this is a upper boundary of growth
        ddbh_maxs = self.PFT.bstem2dbh(self.bstem + self.dbtotal_dt) - self.dbh

        ddbh_ratio_array = np.concatenate(
            (np.arange(0.0,0.991,0.01),
             1. - np.logspace(-2,-6,num=50)))
        biomass_bias = np.zeros((len(ddbh_maxs),len(ddbh_ratio_array)))

        for i_ddbh, ddbh_ratio in enumerate(ddbh_ratio_array):
            ddbh_test = ddbh_maxs * ddbh_ratio
            h_test = self.PFT.dbh2h(self.dbh+ddbh_test)
            btotal_test = self.PFT.size2btotal(self.dbh+ddbh_test,h_test)
            biomass_bias[:,i_ddbh] = np.absolute(
                (btotal_test - btotal) - self.dbtotal_dt)

        # find the best dbh for each size
        best_ddbh_idx = np.argmin(biomass_bias,axis=1).ravel()

        # record the dbh increase
        self.ddbh_dt = ddbh_maxs * ddbh_ratio_array[best_ddbh_idx]

        # calculate all other growth rates
        dbh_new = self.dbh + self.ddbh_dt
        h_new = self.PFT.dbh2h(dbh_new)
        self.dh_dt = h_new - self.h
        self.dbstem_dt = self.PFT.size2bstem(
                            dbh_new,h_new) - self.bstem
        self.dbleaf_dt = self.PFT.size2bleaf(
                            dbh_new,h_new) - self.bleaf
        self.dbroot_dt  = self.dbleaf_dt * self.PFT.q
        self.dbstorage_dt  = self.dbleaf_dt * (1. + self.PFT.q)
        
        self.dba_dt  = np.pi / 4. * (dbh_new) ** 2 - self.ba
        self.dsa_dt  = np.zeros_like(self.dbh) # growth in sapwoo area save for later
        self.dla_dt  = self.dbleaf_dt * self.PFT.SLA

        
        return
        

    def to_csv(self,output_fn=None):
        '''
            Save all the state vars and growth rates into csv file using pandas
        '''

        if output_fn is None:
            output_fn = './growth_size_PFT{:d}.csv'.format(self.PFT.num)

        # create a dictionary
        output_dict = {
            'pft'       : np.ones_like(self.dbh) * self.PFT.num,
            'dbh'       : self.dbh,
            'h'         : self.h,
            'bleaf'     : self.bleaf,
            'bstem'     : self.bstem,
            'broot'     : self.broot,
            'bstorage'  : self.bstorage,
            'ba'        : self.ba,
            'la'        : self.la,
            'sf'        : self.sf,
            'gpp'       : self.gpp,
            'resp_leaf' : self.resp_leaf,
            'resp_stem' : self.resp_stem,
            'resp_root' : self.resp_root,
            'resp_growth' : self.resp_growth,
            'npp'       : self.npp,
            'turnover_leaf' : self.turnover_leaf,
            'turnover_root' : self.turnover_root,
            'turnover_stem' : self.turnover_stem,
            'turnover_bstorage' : self.turnover_bstorage,
            'cb' : self.cb,
            'repro' : self.repro,
            'dbtotal_dt' : self.dbtotal_dt,
            'dbstem_dt' : self.dbstem_dt,
            'dbleaf_dt' : self.dbleaf_dt,
            'dbroot_dt' : self.dbroot_dt,
            'dbstorage_dt' : self.dbstorage_dt,
            'ddbh_dt' : self.ddbh_dt,
            'dh_dt' : self.dh_dt,
            'dba_dt' : self.dba_dt,
            'dla_dt' : self.dla_dt
        }

        output_df = pd.DataFrame(output_dict)
        output_df.to_csv(output_fn,index=False,float_format='%g')


##############################################################################
##############################################################################





