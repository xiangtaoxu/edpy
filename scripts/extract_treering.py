#!usr/bin/env python
# import modules here
import h5py
import numpy as np
import sys
import os
import pandas as pd
import sys


# get the parent folder ABSOLUTE path
# so that we can import ed2_utils
file_path = os.path.abspath(__file__)
path_list = file_path.split('/')
edpy_path = '/'.join(path_list[0:-2])

print(edpy_path)
sys.path.append(edpy_path)


from edpy.extract import extract_treering
import argparse

# default values for input

#####################################################
# parse the argument
parser = argparse.ArgumentParser(
    description='Options for extract tree ring'
)

# add argument to store directory of data input and output
parser.add_argument('--data_dir','-dd',action='store',default='./',type=str)
parser.add_argument('--out_dir','-od',action='store',default='./',type=str)

# add argument to store output prefix
parser.add_argument('--prefix','-p',action='store',default='',type=str)

# add argument to indicate the start and end year/month
# default is None
parser.add_argument('--yeara','-ya',action='store',default=None,type=int)
parser.add_argument('--yearz','-yz',action='store',default=None,type=int)

# end of growth year, default is December
parser.add_argument('--end_of_year','-eoy',action='store',default=12,type=int)


# add argument to indicate which pfts are included
parser.add_argument('--pft','-pft',action='store',default=[],nargs='+',type=int)


# add argument to indicate the smallest tree to core in cm
parser.add_argument('--dbh_min','-dmin',action='store',default=5.,type=float)

# add argument to indicate the minimum hite of a cohort to track
parser.add_argument('--hite_min','-hmin',action='store',default=1.5,type=float)

# Feel free to add more arguments here


# Parse the argument and get its values
args = parser.parse_args()
# the values of each flag can be accessed through
#    args.submit, args.cpu etc.

# conduct some quality check for dir
if args.data_dir[-1] != '/':
    args.data_dr = args.data_dir + '/'

if args.out_dir[-1] != '/':
    args.out_dir = args.out_dir + '/'
    

args = parser.parse_args()

####################################################


data_pf    =  f'{args.data_dir}{args.prefix}'
output_fn  =  f'{args.out_dir}{args.prefix}_tr_{args.yeara}_{args.yearz}.csv'

extract_treering(
     data_pf
    ,output_fn
    ,args.yeara
    ,args.yearz
    ,args.end_of_year
    ,args.pft
    ,args.dbh_min
    ,args.hite_min
)

