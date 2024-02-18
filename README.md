# tax_reforms_political_feasibility
Repo to store codes using the microsimulation tool OpenFisca in order to see whether individuals were winners or losers of past French tax reforms

# Description 

The goal of this repository is to reproduce the framework and American empirical analysis of this paper : https://www.aeaweb.org/articles?id=10.1257/aer.20190021 (Bierbrauer, Felix J., Pierre C. Boyer, and Andreas Peichl. 2021. "Politically Feasible Reforms of Nonlinear Tax Systems." American Economic Review, 111 (1): 153-91.)

The main files of this repositories are the following : 

+ without_reform.py : prepares our dataset (input is a .h5 file, output is a .csv file)

+ utils_paper.py : produces all outputs of the paper (input is the .csv file of the without_reform.py file)

+ useful.sh : contains commands to build the subfolders and launch codes for all years 


# First installation 

Please look at and reproduce step by step installation instructions of this other repository (https://github.com/pvanborre/openfisca_married_couples?tab=readme-ov-file) :
In short, following these instructions will make you convert your input data (SAS files containing ERFS-FPR surveys) to **input datatables** consumables by the microsimulation model OpenFisca. Moreover, these instructions create a **Docker image** called public-openfisca-image that is used to ensure better reproductibility of results (specifies versions for Python and its modules, tags of the OpenFisca repositories). 

Then go to a folder in your computer, open a terminal and clone this repository :
```bash
git clone https://github.com/pvanborre/tax_reforms_political_feasibility.git
```

# All following uses 

First launch Docker Desktop on your computer. If you have errors like "docker daemon is not running", it must be that your app Docker Desktop is not opened in your computer.

Please open a terminal and run the following command.
This command launches your container (entitled openfisca-container) from the Docker image (entitled public-openfisca-image) :
```bash
# Make sure to change paths : for you it should be docker run -it --name openfisca-container -v where_you_cloned/tax_reforms_political_feasibility/codes:/app/codes -v where_you_put_your_data/data:/app/data -v where_you_cloned/tax_reforms_political_feasibility/outputs:/app/outputs public-openfisca-image /bin/bash
docker run -it --name openfisca-container -v C:/Users/pvanb/Projects/tax_reforms_political_feasibility/codes:/app/codes -v C:/Users/pvanb/Projects/openfisca_married_couples/data:/app/data -v C:/Users/pvanb/Projects/tax_reforms_political_feasibility/outputs:/app/outputs public-openfisca-image /bin/bash
# The -v flags indicates that your codes, data and outputs on your disk will be accessible inside the container, in the folders app/codes, app/data and app/outputs
```

From now on make sure to launch **all** your commands from this container : 
you must have something like root@XXXXXXXXXX:/app/codes# 
and you type root@XXXXXXXXXX:/app/codes#  bash useful.sh or root@XXXXXXXXXX:/app/codes# python utils_paper.py -y 2007. 

To launch all codes for all years, please run the following command 
```bash
bash useful.sh
```

If you have these kind of errors (due to the Windows handle of end of files) while launching this command 
```bash
useful.sh: line 2: $'\r': command not found
useful.sh: line 7: $'\r': command not found
useful.sh: line 18: syntax error near unexpected token `$'do\r''
```
just copy paste the content of the file useful.sh in your container (that is your terminal with root@XXXXXXXXXX:/app/codes#).

If you only want to run codes for a specific year (here for instance 2007) :

This first command creates from your input dataset a csv file. Computations done by the microsimulation model are done during this step and are stored in this csv file : 
```bash
python without_reform.py -y 2007
```

Then taking as input this csv file you can produce all outputs of the paper : 
```bash
python utils_paper.py -y 2007
```






