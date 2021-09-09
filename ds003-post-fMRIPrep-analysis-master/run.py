#!/usr/bin/env python3
import sys
import logging
import json
from pathlib import Path
from templateflow.api import get as tpl_get, templates as get_tpl_list
import glob

__version__ = '1.0.0'
logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')


metadata = {
    'Name': 'ds003 example postprocessing',
    'BIDSVersion': '1.1.1',
    'PipelineDescription': {
        'Name': 'ds003-post-fMRIPrep-analysis'
    },
    'CodeURL': 'https://github.com/poldracklab/ds003-post-fMRIPrep-analysis'
}

### parser :
import argparse
opts = argparse.Namespace(output_dir= Path('/storage/albamrt/NMDA/MRI/out'), 
    derivatives_dir = Path('/archive/albamrt/MRI/preprocess/fmriprep'),
    analysis_level = 'first',
    bids_dir = Path('/archive/albamrt/MRI/BIDS'),
    verbose_count = 1,
    ncpus = 4,
    nthreads = 1,
    participant_label = ['E18'],
    # participant_label = [
    #     'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 
    #     'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 
    #     'C22', 'C23', 'C25',
    #     'E07', 'E11', 'E12', 'E13', 'E14', 'E16', 'E18', 
    #     'E20', 'E21', 'E22', 'E24', 'E25',
    #     'E30', 'E31',
    #     'S02', 'S04', 'S05', 'S06', 'S08', 'S11',
    #     'S12', 'S13', 'S16', 'S17', 'S18', 'S21',
    #     'S22', 'S23', 'S24', 'S25'],
    # remove excluded subjects
    group = '*', # * for all
    session = '5', #'1', # * for all
    # participant_label = ['E13', 'E14', 'E16'],
    # run = '1',
    task = ['wm'],
    space = 'MNI152NLin6Asym',
    work_dir = Path('/storage/albamrt/NMDA/MRI/work')
    )

import os
os.chdir('/storage/albamrt/NMDA/MRI/scripts/ds003-post-fMRIPrep-analysis-master')

def get_parser():
    """Define the command line interface"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='DS000003 Analysis Workflow',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        'derivatives_dir', action='store', type=Path,
        help='the root folder of a derivatives set generated with fMRIPrep '
             '(sub-XXXXX folders should be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store', type=Path,
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['participant', 'group'], nargs='+',
                        help='processing stage to be run, "participant" means individual analysis '
                             'and "group" is second level analysis.')

    parser.add_argument('--version', action='version', version=__version__)

    # Options that affect how pyBIDS is configured
    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--participant-label', action='store', type=str,
                        nargs='*', help='process only particular subjects')
    g_bids.add_argument('--task', action='store', type=str, nargs='*',
                        help='select a specific task to be processed')
    g_bids.add_argument('--run', action='store', type=int, nargs='*',
                        help='select a specific run identifier to be processed')
    g_bids.add_argument('--space', action='store', choices=get_tpl_list() + ['T1w', 'template'],
                        help='select a specific space to be processed')
    g_bids.add_argument('--bids-dir', action='store', type=Path,
                        help='point to the BIDS root of the dataset from which the derivatives '
                             'were calculated (in case the derivatives folder is not the default '
                             '(i.e. ``BIDS_root/derivatives``).')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                         help="increases log verbosity for each occurence, debug level is -vvv")
    g_perfm.add_argument('--ncpus', '--nprocs', action='store', type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--nthreads', '--omp-nthreads', action='store', type=int,
                         help='maximum number of threads per-process')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', type=Path,
                         help='path where intermediate results should be stored')

    return parser


def main():
    """Entry point"""
    from os import cpu_count
    from multiprocessing import set_start_method
    from bids.layout import BIDSLayout
    from nipype import logging as nlogging
    import numpy as np
    import pandas as pd
    set_start_method('forkserver')
#    opts = get_parser().parse_args()

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    # Resource management options
    plugin_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {
            'n_procs': opts.ncpus,
            'raise_insufficient': False,
            'maxtasksperchild': 1,
            'memory_gb':15
        }
    }
    # Permit overriding plugin config with specific CLI options
    if not opts.ncpus or opts.ncpus < 1:
        plugin_settings['plugin_args']['n_procs'] = cpu_count()

    nthreads = opts.nthreads
    if not nthreads or nthreads < 1:
        nthreads = cpu_count()

    derivatives_dir = opts.derivatives_dir.resolve()
    bids_dir = opts.bids_dir or derivatives_dir.parent

    # Get absolute path to BIDS directory
    bids_dir = opts.bids_dir.resolve()
    layout = BIDSLayout(str(bids_dir), validate=False, derivatives=str(derivatives_dir))
    query = {'domains': 'derivatives', 'desc': 'preproc',
             'suffix': 'bold', 'extension': ['.nii', '.nii.gz']}

    if opts.participant_label:
        #query['subject'] = '|'.join(opts.participant_label)
        query['subject'] = opts.participant_label
    # if opts.run:
    #      query['run'] = '|'.join(opts.run)
    if opts.task:
        query['task'] = '|'.join(opts.task)
    if opts.space:
        query['space'] = opts.space
        if opts.space == 'template':
            query['space'] = '|'.join(get_tpl_list())


    # Preprocessed files that are input to the workflow
    prepped_bold = layout.get(**query)
    if not prepped_bold:
        print('No preprocessed files found under the given derivatives folder')

    prepped_bold = [prep for prep in prepped_bold if 'ses-0' + opts.session in prep.filename]
  #  prepped_bold = [prep for prep in prepped_bold if 'run-0' + opts.run in prep.filename]


    # The magic happens here
    if 'first' in opts.analysis_level:
        from workflows import first_level_wf

        output_dir = opts.output_dir.resolve()
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info('Writting 1st level outputs to "%s".', output_dir)
        base_entities = set(['subject', 'session', 'task', 'run', 'acquisition', 'reconstruction'])
        inputs = {}
        for part in prepped_bold:
            entities = part.entities
            sub = entities['subject']
            ses = entities['session']
            if 'run' in entities.keys():
                run = entities['run']
            else:
                run = 1
            inputs = {}
            base = base_entities.intersection(entities)
            subquery = {k: v for k, v in entities.items() if k in base}
            inputs['bold'] = part.path
            inputs['mask'] = layout.get(
                domains='derivatives',
                suffix='mask',
                return_type='file',
                extension=['.nii', '.nii.gz'],
                space=query['space'],
                **subquery)[0]
            inputs['events'] = layout.get(
                suffix='events', return_type='file', **subquery)[0]
            inputs['regressors'] = layout.get(
                domains='derivatives',
                suffix='regressors',
                return_type='file',
                extension=['.tsv'],
                **subquery)[0]
            #inputs['tr'] = part.entities['RepetitionTime']
            inputs['tr'] = 0.745

            id = sub+ses+str(run)
            workflow = first_level_wf(in_files = inputs, output_dir = output_dir, name = id)
            workflow.base_dir = opts.work_dir / 'wf_1st_level'
            workflow.run(**plugin_settings)


    if 'second' in opts.analysis_level:
        from workflows import second_level_wf
        import re

        output_dir = opts.output_dir.resolve()
        metafile = '{}/1st_level/dataset_description.json'.format(output_dir)
        with open(metafile, 'w') as metafile:
            json.dump(metadata, metafile, indent=4)
        glayout = BIDSLayout(str(bids_dir), validate=False, derivatives=str(output_dir/'1st_level/'))

        contrasts = ['visual', 'delay', 'response', 'memory', 'recall']

        for i in range(0,len(contrasts)): # number of copes (contrasts in first level)
            base_entities = set(['subject', 'session', 'task', 'run', 'acquisition', 'reconstruction'])
            in_copes = []
            in_varcopes = []
            ids = []
            out_containers = []
            for part in prepped_bold:
                entities = part.entities
                base = base_entities.intersection(entities)
                subquery = {k: v for k, v in entities.items() if k in base}
                in_copes.append([f for f in glayout.get(
                    domains='all',
                    suffix=contrasts[i],
                    return_type='file',
                    extension=['.nii', '.nii.gz'],
                    space=query['space'],
                    **subquery) if '_cope' in f][0])
                in_varcopes.append([f for f in glayout.get(
                    domains='all',
                    suffix=contrasts[i],
                    return_type='file',
                    extension=['.nii', '.nii.gz'],
                    space=query['space'],
                    **subquery) if '_varcope' in f][0])
                entities = part.entities
                ids.append(entities['subject'] + entities['session'])
                out_containers.append('sub-' + entities['subject'] + '/ses-' + entities['session'])


            out_containers = np.unique(out_containers).tolist()

            bids_ref = re.sub('sub-.[0-5][0-9]+', 'sub-all', prepped_bold[0].path)

            group_mask = tpl_get(entities['space'],
                                 resolution=2,
                                 desc='brain',
                                 suffix='mask')
            aux = contrasts[i]
            group_out = output_dir / '2nd_level' / aux
            group_out.mkdir(exist_ok=True, parents=True)

            workflow = second_level_wf(group_out, bids_ref)

            regress_df = pd.DataFrame(ids, columns=['subj']) 
            regress_mat = pd.get_dummies(regress_df['subj'])  
            regressors_l2 = regress_mat.to_dict('list')  

            reg = np.unique(ids).tolist()
            contrast_mat = np.diag(np.full(len(reg), 1))
            contrasts_l2 = []
            for i, s in enumerate(reg):
                contrasts_l2.append([s, 'T', reg, contrast_mat[i].tolist()])


            # set inputs
            workflow.inputs.inputnode.group_mask = str(group_mask)
            workflow.inputs.inputnode.in_copes = in_copes
            workflow.inputs.inputnode.in_varcopes = in_varcopes
            workflow.inputs.inputnode.contrasts = contrasts_l2
            workflow.inputs.inputnode.regressors = regressors_l2
            workflow.inputs.inputnode.out_containers = out_containers
            workflow.base_dir = opts.work_dir
            workflow.run(**plugin_settings)


    if 'third' in opts.analysis_level:
        from workflows import third_level_wf
        import re
        contrasts = ['visual', 'delay', 'response', 'memory', 'recall']

        if opts.group != '*':
            prepped_bold = [prep for prep in prepped_bold if 'sub-' + opts.group in prep.filename]
        if opts.session != '*':
            prepped_bold = [prep for prep in prepped_bold if 'ses-0' + opts.session in prep.filename]
        
        for i in range(0, len(contrasts)):
            if opts.group == '*':
                aux = 'all' + '_' + contrasts[i]
            else:
                aux = opts.group + '_' + contrasts[i]
            output_dir = opts.output_dir.resolve() / '2nd_level' / contrasts[i]

            metafile = '{}/dataset_description.json'.format(output_dir)
            with open(metafile, 'w') as metafile:
                json.dump(metadata, metafile, indent=4)
            glayout = BIDSLayout(str(bids_dir), validate=False, derivatives=str(output_dir))
            base_entities = set(['subject', 'session', 'task', 'run', 'acquisition', 'reconstruction'])

            in_copes = []
            in_varcopes = []
            entities = prepped_bold[0].entities
            # for part in prepped_bold:
            #     entities = part.entities
            #     base = base_entities.intersection(entities)
            #     subquery = {k: v for k, v in entities.items() if k in base}
            #     in_copes.append(glayout.get(
            #         domains='derivatives',
            #         suffix='cope',
            #         return_type='file',
            #         extension=['.nii', '.nii.gz'],
            #         space=query['space'],
            #         **subquery)[0])
            #     in_varcopes.append(glayout.get(
            #         domains='derivatives',
            #         suffix='varcope',
            #         return_type='file',
            #         extension=['.nii', '.nii.gz'],
            #         space=query['space'],
            #         **subquery)[0])

            in_copes = glob.glob(str(output_dir) + '/sub-' + opts.group + '*/ses-0' + opts.session + '/*/cope*.nii.gz')
            in_varcopes = glob.glob(str(output_dir) + '/sub-' + opts.group + '*/ses-0' + opts.session + '/*/varcope*.nii.gz')
            bids_ref = re.sub('sub-.[0-5][0-9]+', 'sub-all', prepped_bold[0].path)
                                                
            group_mask = tpl_get(entities['space'],
                                 resolution=2,
                                 desc='brain',
                                 suffix='mask')

            group_out = opts.output_dir.resolve() / '3rd_level' / aux
            group_out.mkdir(exist_ok=True, parents=True)

            workflow = third_level_wf(group_out, bids_ref)

            # set inputs
            workflow.inputs.inputnode.group_mask = str(group_mask)
            workflow.inputs.inputnode.in_copes = in_copes
            workflow.inputs.inputnode.in_varcopes = in_varcopes

            workflow.base_dir = opts.work_dir
            workflow.run(**plugin_settings)

    return 0


if __name__ == '__main__':
    sys.exit(main())

