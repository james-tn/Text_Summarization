#!/bin/bash -x

# Usage:  ./setup.sh [output_dir]
#   First arg is output directory, eg: ./setup.sh ../output
#   All other args are passed straight through to create_data.py

[ ! -f util/create_data.py ] &&
    { echo "** error: run $0 from 'code' directory."; exit 2; }

# if first arg doesn't start with a dash (--foo), assume output dir
[ $# -gt 0 -a "${1:0:1}" != '-' ] &&
    default_args="--out_dir $1" &&
    shift

# python script takes args in format: --some_arg [some_value]
# To show usage: python3 util/create_data.py -h
python3 util/create_data.py $default_args $@

