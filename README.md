# Instructions of usage

## Preprocessing
In order to build our GLM we first need to preprocess the MRI files. To do so we use [fMRIPrep](https://fmriprep.org/en/stable/). More instructions on how to install the needed software, prepare files & run fmriprep [here](https://docs.google.com/document/d/1L1_kZFeQnlUSomfed6QUyLGLpRdGokajbXyriOP5m_Y/edit?usp=sharing) or in the file 'InstallationAndUsage.pdf'.
* Before using fmriprep we need to convert our raw DCM files (/archive/hstein/RawData/RawMRI) to BIDS format (/archive/albamrt/MRI/BIDS). To do so we use [dcm2bids](https://github.com/UNFmontreal/Dcm2Bids). 
0. Before using it, delete sbref volumes, leaving just one of them. To do so, if there is a folder in the DCM files with 10 files, just delete all of them but the first.
1. After installing it according to the [instructions](https://docs.google.com/document/d/1L1_kZFeQnlUSomfed6QUyLGLpRdGokajbXyriOP5m_Y/edit?usp=sharing), open the console and go to the directory where you want to export the BIDS files and type: 
```
dcm2bids -d <DCM directory> -p <participantID> -s <sessionID> -c <document.json directory>
```
 e.g. for subject C20, session 1 this would be: 
 ```
 dcm2bids -d /archive/hstein/RawData/RawMRI/C20/1/FILESET/C20/0 -p C20 -s 01 -c /archive/albamrt/MRI/document.json 
 ```
2. Go to the new folder with the files in BIDS format and open the subfolder called 'func'. Copy the files 'sub-C20_ses-01_task-wm_run-01_sbref.json' and 'sub-C20_ses-01_task-wm_run-01_sbref.nii.gz' and rename them to 'sub-C20_ses-01_task-wm_run-02_sbref.json' and 'sub-C20_ses-01_task-wm_run-02_sbref.nii.gz'.
3. In the subjects' BIDS directory, go to the 'fmap' folder. Open the file 'sub-C20_ses-01_phasediff.json' and change the first line to:
```
{
    "IntendedFor": [
        "ses-01/func/sub-E18_ses-01_task-wm_run-01_bold.nii.gz",
        "ses-01/func/sub-E18_ses-01_task-wm_run-02_bold.nii.gz",
        "ses-01/func/sub-E18_ses-01_task-wm_run-01_sbref.nii.gz",
        "ses-01/func/sub-E18_ses-01_task-wm_run-02_sbref.nii.gz"],
```
* We can now use fmriprep. To do so we just need to type on the console (substituting C20 for the name of the subject):
```
docker run --rm -it -e DOCKER_VERSION_8395080871=18.09.2 -v /storage/albamrt/NMDA/MRI/license.txt:/opt/freesurfer/license.txt:ro -v /storage/albamrt/NMDA/MRI/BIDS:/data:ro -v /storage/albamrt/NMDA/MRI/preprocess:/out -v /storage/albamrt/NMDA/MRI/work:/scratch  -u $UID poldracklab/fmriprep:1.4.0 /data /out participant --participant_label C20 -t wm --ignore slicetiming --bold2t1w-dof 6 --no-submm-recon --fs-no-reconall --write-graph -v --output-spaces MNI152NLin6Asym:res-2 --write-graph --n_cpus 15 --omp-nthreads 4 --nthreads 4 --mem-mb 20000 -w /scratch --low-mem
```
## Creating the tsv file
In order to run the GLM we need a .tsv file with the regressors. We can build this file with the script 'create_tsv.py', that takes the behavior files of the speified subjects in '/archive/albamrt/MRI/behaviour/', converts them to .tsv format and saves them in '/archive/albamrt/NMDA/MRI/BIDS/'.

## GLM
For the GLM we have adapted the code in https://github.com/poldracklab/ds003-post-fMRIPrep-analysis. The adapted scripts are in the folder 'ds003-post-fMRIPrep-analysis-master'. There are three steps:
* First level: Running the GLM for each of the two runs of the task for each subject and session. We just need to modify some parameters in the script 'run.py'. For example, to run it for participant C20, session 1 we would just set:
```
    analysis_level = 'first',
    participant_label = ['C20'],
    group = '*',
    session = '1', 
```
* Second level: The second level averages the results of the first level so that we end up having results for each subject and session and not on the run level anymore. To run the second level for subject C20, session1 we would just have to modify the script 'run.py' again to:
```
    analysis_level = 'second',
    participant_label = ['C20'],
    group = '*',
    session = '1', 
```
* Third level: Finally, the third level averages our second level files to obtain group contrasts. To do so we just need to modify the run.py file again. So, for example, if we wanted to average the files from the first session of subjects C20, C21, C22 we would do:
```
    analysis_level = 'third',
    participant_label = ['C20', 'C21', 'C22'],
    group = 'C',
    session = '1', 
```
Another option is just to set the participant label to all the target participants and then filter by the group one is interested in. Taht would be:
```
    analysis_level = 'third',
    participant_label = ['C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 
        'C22', 'C23', 'C25',
        'E07', 'E11', 'E12', 'E13', 'E14', 'E16', 'E18', 
        'E20', 'E21', 'E22', 'E24', 'E25',
        'E30', 'E31',
        'S02', 'S04', 'S05', 'S06', 'S08', 'S11',
        'S12', 'S13', 'S16', 'S17', 'S18', 'S21',
        'S22', 'S23', 'S24', 'S25'],
    group = 'C',
    session = '1', 
```
