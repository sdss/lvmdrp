# lvmdrp
Local Volume Mapper (LVM) Data Reduction Pipeline

The LVM DRP in it's current incarnation installs a collection of routines which make use of the DESI-DRP, specifically spectre and DESI-PREPROC.

The purpose of the current repository is 2 fold. 

1. Automatically recreate the DESI environment, with dependencies and required  
2. Host LVM routines/wrappers/patches to operate the based DESI-DRP


## Installation
Installation scripts are contained in the lvmdrp/utils directory.
Installation is based around using an Anaconda development environment. 
1. (If necessary) create an anaconda environment named lvmdpr
> conda crate --name lvmdrp python=3.8
2. repo_update.sh is used to clone, clean and updated repositories
> repo_update.sh --reclone : pull all dependent repositories.
> repo_update.sh --update  : pull changes from dependent repositories.
> repo_update.sh --clean   : Delete all dependent repositories.

## Creating Test Data