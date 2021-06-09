#!/bin/bash
#
# Desc: Script to collect data for vector extraction and clustering
# Auth: Mark Swaringen
#
# Copyright: Aigritec 2021
##################################################

# For printing:
# Red is 1
# Green is 2
# Reset is sgr0

# ENSURE THIS VARIABLE IS THE ROOT DIRECTORY
ROOT="perception"

###################
# CHECKS          #
###################

# check if we are in the correct directory (should be the root dir)
# if [ "${PWD##*/}" != "$ROOT" ]; then

#     tput setaf 1
#     printf "\n"
#     printf " - Please run this script from the perception root directory, using the following command: \n"
#     printf "\n"
#     printf "          sh scripts/collect_data.sh \n"
#     printf "\n"
#     tput sgr0

#     # stop the script
#     exit 1

# fi

# create apple data folder
mkdir -p data/minneapple

###################
# GET DATA        #
###################

# check if the apples directory is empty or not
if [ -z "$(ls -A data/minneapple)" ]; then
    wget -O data/minneapple/detection.tar.gz https://conservancy.umn.edu/bitstream/handle/11299/206575/detection.tar.gz?sequence=2\&isAllowed=y

    # let user know file has been downloaded and will be extracted
    tput setaf 2
    printf "\n"
    printf " - Downloaded data file \n"
    printf " - Extracting data... \n"
    tput sgr0

    # unpack the downloaded file and move it to the correct folder
    tar -xzf data/minneapple/detection.tar.gz --checkpoint=.1000
    printf "\n"

    # alter the file structure to our liking
    mv detection/train data/minneapple/train
    mv detection/test data/minneapple/test

    # remove used files
    rm -r detection
    rm data/minneapple/detection.tar.gz

    mkdir -p data/minneapple/vectors

else
    tput setaf 2
    printf "\n"
    printf " - Data already collected \n"
    printf "\n"
    tput sgr0
fi