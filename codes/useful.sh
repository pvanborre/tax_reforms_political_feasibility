#!/bin/bash



################################################################################
# Ensures we are at the good location to launch codes 
expected_path="/app/codes"
current_path=$(pwd)

if [ "$current_path" != "$expected_path" ]; then
    echo "Error: Please run this script from the 'codes' folder."
    exit 1
fi
################################################################################

cd excel 

################################################################################
# For the first setup, creates folder to store excel files
for year in {2002..2018}; do
    next_year=$((year + 1))
    folder_name="${year}-${next_year}"

    # Create the folder if it doesn't exist
    if [ ! -d "$folder_name" ]; then
        mkdir "$folder_name"
        echo "Created folder: $folder_name"
    else
        echo "Folder already exists: $folder_name"
    fi
done
################################################################################

cd ..

################################################################################
# Get data from the simulations
for i in $(seq 2002 2018);
do
    python without_reform.py -y $i
done
################################################################################

################################################################################
# Run utils
for i in $(seq 2002 2018);
do
    python utils_paper.py -y $i
done
################################################################################

