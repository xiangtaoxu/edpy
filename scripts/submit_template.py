# import packages
# Notes: I am trying to only import standard packages in python3
#        so that it can be used for most linux OS

import subprocess # for final system calls
import fileinput # for modifying ED2IN
import xml.etree.ElementTree as xmlET
import xml.dom.minidom as minidom
import argparse # for processing arguments of this script
from pathlib import Path  # deal with path/directories
import time # use the sleep function
import os # use to get current working directory


################################################
# Process script argument
################################################

parser = argparse.ArgumentParser(
    description='Options for submitting ED2 simulations')

# add argument to indicate whether to submit the simulation
# this is an optional argument thus starts with a -,
# The script will ONLY submit the run when -s or --submit is included.
parser.add_argument('--submit','-s',action='store_true')

# add argument to indicate how many cpu cores to use
# by default it uses 20 cores
parser.add_argument('--cpu','-c',action='store',default=20)

# add argument to indicate the total simulation wall time
# default is 1000 hours
parser.add_argument('--sim_time','-t',action='store',default="1000:00:00")

# Feel free to add more arguments here
# e.g. output directory etc.


# Parse the argument and get its values
args = parser.parse_args()
# the values of each flag can be accessed through
#    args.submit, args.cpu etc.

#################################################




#################################################
# Define some constants and configuration
#################################################
# I/O
work_dir   = os.getcwd() + '/'
output_dir = "/ibstorage/xiangtao/ED2_treering/HKK_spinup/"
Path(output_dir).mkdir(parents=True, exist_ok=True)
# create the directory if no already present
# parents=True indicates recursively create any non-existant parent folders
# exist_ok indicates not to throw an error if the folder is already existant.

# ed2 executable to use
ed2_exec = work_dir + 'ed2'

#----------------------------------------------
# ED2IN information
#----------------------------------------------
# ED2IN template to copy from
ED2IN_template = "/home/xx286/ED2/ED/run/ED2IN"

# use a dictionary to store all ED2IN flags to be changed
ED2IN_flags_common = {}

# assign flags that are same for all simulations

# runtype and initilization
ED2IN_flags_common['RUNTYPE'] = "'INITIAL'"
ED2IN_flags_common['IED_INIT_MODE'] = '0'

# start and end time time
ED2IN_flags_common.update({
        'IMONTHA' : '07',
        'IDATEA' : '01',
        'IYEARA' : '1600',
        'ITIMEA' : '0000',
        'IMONTHZ' : '07',
        'IDATEZ' : '01',
        'IYEARZ' : '1850',
        'ITIMEZ' : '0000',
    })

# I/O options
ED2IN_flags_common.update({
        'ISOUTPUT' : '3',
        'IMOUTPUT' : '3',
        'IQOUTPUT' : '3',
        'IYOUTPUT' : '3',
        'ITOUTPUT' : '0',
        'IDOUTPUT' : '0',
        'IFOUTPUT' : '0',
        'UNITSTATE' : '3',
        'FRQSTATE' : '1',
})


# Site information
ED2IN_flags_common['POI_LAT'] = "15.6"
ED2IN_flags_common['POI_LON'] = "99.2"

ED2IN_flags_common['SLXCLAY'] = "1."
ED2IN_flags_common['SLXSAND'] = "1."
ED2IN_flags_common['NSLCON'] = "3"
# clay and sand fraction unknown
# use sandy load predefined in the model

ED2IN_flags_common['NZG'] = "16" # 16 soil layers
ED2IN_flags_common['SLZ'] = "-8.00,-6.50,-5.50,-4.50,-3.50,-3.00,-2.50,-2.00,-1.50,-1.00,-0.80,-0.60,-0.40,-0.30,-0.20,-0.10"

# necessary external data bases
ED2IN_flags_common.update({
        'VEG_DATABASE' : "'/ibstorage/shared/ed2_data/oge2OLD/OGE2_'",
        'SOIL_DATABASE' : "'/ibstorage/shared/ed2_data/soil_ed22/Quesada+RADAM+IGBP/Quesada_RADAM_IGBP_'",
        'LU_DATABASE' : "'/ibstorage/shared/ed2_data/land_use/glu+sa1/glu+sa1-'",
        'THSUMS_DATABASE' : "'/ibstorage/shared/ed2_data/ed_inputs/'",
})

# MET HEADER and information
ED2IN_flags_common.update(
    {
        'ED_MET_DRIVER_DB' : "'/ibstorage/shared/ED_MET/HKK/HKK_corrected_HEADER'",
        'METCYC1'       : '1901',
        'METCYCF'       : '2000',
        'IMETAVG'       : '2',
        'IMETRAD'       : '0',
        'INITIAL_CO2'   : '285.',
    }
)

# physiological parameters
ED2IN_flags_common.update(
    {
        'IALLOM' : '4',
        'ECONOMICS_SCHEME' : '1',
        'IGRASS' : '0',
        'H2O_PLANT_LIM' : '4',
        'PLANT_HYDRO_SCHEME' : '1',
        'ISTRUCT_GROWTH_SCHEME' : '2',
        'ISTOMATA_SCHEME' : '1', # katul's model
        'CARBON_MORTALITY_SCHEME' : '2', # growth-based carbon mortlaity
        'HYDRAULIC_MORTALITY_SCHEME' : '0', # no hydraulic mortality
        'GROWTHRESP' : '0.333',
        'Q10_C3' : '2.11',
        'Q10_C4' : '2.11',
        'INCLUDE_THESE_PFT' : "1,2,3,4", # grass +  3 tropical PFTs
        'MAXPATCH' : '20',
        'MAXCOHORT' : '40',
        'TREEFALL_DISTURBANCE_RATE' : '0.005',  # specific to HKK
        'IPHEN_SCHEME' : '4', # hydraulics-drivne phenology
        'REPRO_SCHEME' : '3', # continuous function for reproduction
    }
)


# mischellaneous
ED2IN_flags_common['YR1ST_CENSUS'] = 3200 # maximum possible values, do not include census

#----------------------------------------------
################################################



################################################
# Job preparation and  submission
################################################

# we submit two spinups with trait plasticity turned on and off
TPS_vals=['0', '3']
# we submit two spinups with hydraulics-driven phenology
# but with different metric
# 5 uses solely dmax_leaf_psi
# 6 uses both dmax and dmin psi
# 6 tends to yield more deciduous results
Pheno_vals = [5,6] 

# create simulation array
sim_array = []
for i, TPS in enumerate(TPS_vals):
    for j, Pheno in enumerate(Pheno_vals):
        sim_array.append((TPS,Pheno))

##################################################################
# loop over the values and submit one simulation for each value
# !!!! WARNING:
#    Modify ED2IN_flags_run (not ED2IN_flags_common) within the loop
###################################################################

for sim_setup in sim_array:
    # disaggregate the sim_setup
    TPS = sim_setup[0]
    Pheno = sim_setup[1]

    # copy the common ED2IN flags
    ED2IN_flags_run = ED2IN_flags_common.copy()

    # update flags specific to this simulations
    ED2IN_flags_run['TRAIT_PLASTICITY_SCHEME'] = TPS

    job_name = f"HKK_spinup_TPS{TPS}_Ph{Pheno}"
    ED2IN_fn = work_dir + f'ED2IN_{job_name}'

    ED2IN_flags_run['FFILOUT'] = f"'{output_dir}/{job_name}'"
    ED2IN_flags_run['SFILOUT'] = f"'{output_dir}/{job_name}'"


    # xml
    # create and set xml
    xml_name = f"{work_dir}{job_name}.xml"
    with open(xml_name,'w') as f:
        # create the root layer config
        xml_config = xmlET.Element('config')
        xml_tree = xmlET.ElementTree(xml_config)

        # for each pft create a dictionary
        # halve the mortality
        pfts = [
                {'num' : 2, 'rho' : 0.50, 'b1Ht' : 1.6712, 'b2Ht' : 0.4429, 
                'phenology' : Pheno, 'mort3' : 0.0095},
                {'num' : 3, 'rho' : 0.65, 'b1Ht' : 1.5815, 'b2Ht' : 0.4429,
                'phenology' : Pheno, 'mort3' : 0.0048},
                {'num' : 4, 'rho' : 0.80, 'b1Ht' : 1.5105, 'b2Ht' : 0.4429,
                'phenology' : Pheno, 'mort3' : 0.0027},
            ]
        for pft in pfts:
            pft_xml = xmlET.SubElement(xml_config,'pft')
            for var in pft.keys():
                var_xml = xmlET.SubElement(pft_xml,var)
                var_xml.text = f"{pft[var]}"

        xml_str_pretty = minidom.parseString(xmlET.tostring(xml_tree.getroot(),encoding='unicode')).toprettyxml()

        f.write(xml_str_pretty)

        
    ED2IN_flags_run['IEDCNFGF'] = f"'{xml_name}'"# xml file to use



    #---------------------------------------------
    # create and modify ED2IN
    #---------------------------------------------
    subprocess.run([f'cp {ED2IN_template} {ED2IN_fn}'],shell=True)

    with fileinput.FileInput(ED2IN_fn, inplace=True, backup='.bak') as file:
        for line in file:
            # only update the lines start with NL
            line_no_ws = line.lstrip() # get rid of th leading white space
            num_of_space = len(line) - len(line_no_ws)
            if len(line_no_ws) < 2 or line_no_ws[0:2] != 'NL':
                print(line,end='') # print back the original line
                continue

            # this is a flag line, check whether it contains any of our key in the ED2IN_flags_run
            # get flag name, the first three character is NL%
            flag = line_no_ws.split(' ')[0][3::] # 

            # if present, replace the line
            if flag in ED2IN_flags_run.keys():
                line = " "*num_of_space + f"NL%{flag} = {ED2IN_flags_run[flag]}\n"
            # print back the content
            print(line,end='')
    

    #---------------------------------------------
    # submit job
    #---------------------------------------------

    # command string for sbatch
    # TODO: add quick post-processing
    cmd_strs = [
        f"ulimit -s unlimited",
        f"export OMP_NUM_THREADS={args.cpu}",
        f"cd \$SLURM_SUBMIT_DIR",
        f"srun -n 1 --cpus-per-task={args.cpu} {ed2_exec} -f {ED2IN_fn}"
    ]
    cmd_str = " ; ".join(cmd_strs)

    #option flags for sbatch
    slurm_opts = [
        f"-o {job_name}.out",
        f"-e {job_name}.err",
        f"-J {job_name}",
        f"-t {args.sim_time}",
        f"--mem-per-cpu=1000",
        f"-n 1",
        f"-c {args.cpu}",
        f"--mail-type=END",
        f"--mail-user=NULL",
    ]
    slurm_opt = ' '.join(slurm_opts)

    print(slurm_opt)
    print(cmd_str)

    # submit job
    if args.submit:
        # this is compatible with python <3.6
        result = subprocess.run(
            f'sbatch {slurm_opt} --wrap="{cmd_str}"',
            stdout = subprocess.PIPE,shell=True)
        # print the return value (jobid) of the submission
        print(result.stdout,end='')

        # wait for at least 0.1 seconds before the next submission
        # some platform will block too frequent job submissions
        time.sleep(0.1)
    else:
        print("Only a test; Job not submitted. Use -s to submit the jobs")


