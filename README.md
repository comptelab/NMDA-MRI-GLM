# Instructions of usage

## Preprocessing
In order to build our GLM we first need to preprocess the MRI files. To do so we use [fMRIPrep](https://fmriprep.org/en/stable/). More instructions on how to install the needed software, prepare files & run fmriprep [here](https://docs.google.com/document/d/1L1_kZFeQnlUSomfed6QUyLGLpRdGokajbXyriOP5m_Y/edit?usp=sharing).
* We need to convert our raw DCM files (/archive/hstein/RawData/RawMRI) to BIDS format (/archive/albamrt/MRI/BIDS). To do so we use [dcm2bids](https://github.com/UNFmontreal/Dcm2Bids). 
0. Before using it, delete sbref volumes, leaving just one of them. To do so, if there is a folder in the DCM files with 10 files, just delete all of them but the first.) 
1. After installing it, open the console and go to the directory where you want to export the BIDS files and type: 
```
dcm2bids -d <DCM directory> -p <participantID> -s <sessionID> -c <document.json directory>
```
 e.g. for subject C08, session 1 whits would be: 
 ```
 dcm2bids -d /archive/hstein/RawData/RawMRI/C08/1/FILESET/C08/0 -p C08 -s 01 -c /archive/albamrt/MRI/document.json 
 ```
2. 
