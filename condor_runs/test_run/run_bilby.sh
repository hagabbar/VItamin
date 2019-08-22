#!/bin/bash

# Shell script to run bilby parameter estimation codes via condor
../../bilby_pe.py -label=${1} -od=${2}
