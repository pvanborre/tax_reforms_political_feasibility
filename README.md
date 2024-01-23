# tax_reforms_political_feasibility
Repo to store codes using the microsimulation tool OpenFisca in order to see whether individuals were winners or losers of past French tax reforms

# Description 

The goal of this repository is to reproduce the framework and American empirical analysis of this paper : https://www.aeaweb.org/articles?id=10.1257/aer.20190021 (Bierbrauer, Felix J., Pierre C. Boyer, and Andreas Peichl. 2021. "Politically Feasible Reforms of Nonlinear Tax Systems." American Economic Review, 111 (1): 153-91.)

The main files of this repositories are the following : 
TODO 


# Installation 

Please look at the installation instructions of this other repository (https://github.com/pvanborre/openfisca_married_couples?tab=readme-ov-file) :
In short, following these instructions will make you convert your input data (SAS files containing ERFS-FPR surveys) to input datatables consumables by the microsimulation model OpenFisca. Moreover, these instructions create a Docker image that is used to ensure better reproductibility of results (specifies versions for Python and its modules, tags of the OpenFisca repositories). 

To run codes, first launch Docker Desktop on your computer. 

Only for the first use : go to a folder in your computer, open a terminal, clone this repository and go into it :
```bash
git clone https://github.com/pvanborre/tax_reforms_political_feasibility.git
cd tax_reforms_political_feasibility
```

For all uses : please launch a terminal and go to the tax_reforms_political_feasibility folder.

Then launch your container from the Docker image :
```bash
# replace my 3 paths by where_you_cloned/tax_reforms_political_feasibility/codes and where_you_put_your_data/data and where_you_cloned/tax_reforms_political_feasibility/outputs
docker run -it --name openfisca-container -v C:/Users/pvanb/Projects/tax_reforms_political_feasibility/codes:/app/codes -v C:/Users/pvanb/Projects/openfisca_married_couples/data:/app/data -v C:/Users/pvanb/Projects/tax_reforms_political_feasibility/outputs:/app/outputs public-openfisca-image /bin/bash
# The -v flags indicates that your codes, data and outputs on your disk will be accessible inside the container, in the folders app/codes, app/data and app/outputs
```






