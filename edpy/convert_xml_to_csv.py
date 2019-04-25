#!usr/bin/env python

# This script converts xml files generated from ED2 to more readable csv file
# to use it just type
# python convert_xml_to_csv.py XXX.xml

# modules
import numpy as np
import pandas as pd
from lxml import etree
import sys


# first get filename

file_name_raw = str(sys.argv[1])

# process the string to get file_name without path and suffix
file_name = file_name_raw.split('/')[-1]  # get rid of path
file_name = file_name.split('.xml')[0]

# create structure for output
output_filename = './{:s}.csv'.format(file_name)

output_dict = {}

# list of variables to extract
infor_params = ['num','is_tropical','is_grass','is_liana']
alloc_params = ['SLA','rho','q','qsw','agf_bs','SRA','leaf_turnover_rate','root_turnover_rate'] 
photo_params = ['Vm0','dark_respiration_factor','root_respiration_factor','stem_respiration_factor']
hydro_params = ['leaf_psi_tlp','wood_Kmax','wood_psi50']
allom_params = ['b1Ht','b2Ht','hgt_min','hgt_max','repro_min_h',
                'b1Bl_small','b2Bl_small','b1Bs_small','b2Bs_small',
                'b1SA','b2SA','b1Rd','b2Rd','root_beta']
mort_params = ['mort0','mort1','mort2','mort3',
               'mort_alpha','mort_beta',
               'mort_plc_max','mort_plc_th',
               'seedling_mortality']
other_params = ['init_density','seed_rain']

output_params = (infor_params + alloc_params + photo_params + hydro_params + allom_params +
                 mort_params +other_params)

output_dict['Parameter'] = output_params

#read in the xml
params_xml = etree.parse(file_name_raw)
xml_root = params_xml.getroot()

# loop over xml_root to find pft
for root_element in xml_root.iter():
    if root_element.tag == 'pft':
        # we need to create a structure for pft
        pft_params = np.zeros((len(output_params),))

        # loop over element in root_element
        for trait_element in root_element.iter():
            if trait_element.tag in output_params:
                # find out which one it is
                trait_idx = np.where(np.array(output_params) == trait_element.tag)[0][0]
                pft_params[trait_idx] = np.float(trait_element.text)

        # add pft_params to the output_dict
        output_dict['PFT {:d}'.format(int(pft_params[0]))] = pft_params.tolist()

# Now we construct pandas dataframe
output_df = pd.DataFrame(output_dict,index=output_params)
# save
output_df.to_csv(output_filename,index=False)
