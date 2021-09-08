# Instructions of usage

## Preprocessing
In order to build our GLM we first need to preprocess the MRI files. To do so we use [fMRIPrep](https://fmriprep.org/en/stable/). More instructions on how to install the needed software, prepare files & run fmriprep [here](https://docs.google.com/document/d/1L1_kZFeQnlUSomfed6QUyLGLpRdGokajbXyriOP5m_Y/edit?usp=sharing).
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
3. In the subjects' bIDS directory, go to the 'fmap' folder. Open the file 'sub-C20_ses-01_phasediff.json' and add the follwing lines at the beggining:
```

    "IntendedFor": [
        "ses-01/func/sub-E18_ses-01_task-wm_run-01_bold.nii.gz",
        "ses-01/func/sub-E18_ses-01_task-wm_run-02_bold.nii.gz",
        "ses-01/func/sub-E18_ses-01_task-wm_run-01_sbref.nii.gz",
        "ses-01/func/sub-E18_ses-01_task-wm_run-02_sbref.nii.gz"],
```
