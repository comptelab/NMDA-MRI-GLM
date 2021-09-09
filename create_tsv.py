from nipype import config
config.enable_debug_mode()
from nipype.pipeline.engine import Workflow, Node, MapNode
import nipype.interfaces.utility as util


##############################################################################
# 							FILE SYSTEM SPECIFICATIONS						 #
##############################################################################

# location of data folder and subject identifier
bids_dir = u'/archive/albamrt/MRI/BIDS/'
beh_dir = u'/archive/albamrt/MRI/behaviour/'

subject_id = ['E02']
session_id = '5'

info = Node(interface = util.IdentityInterface(fields = 
	['subject_id', 'session_id']), name = 'info')
# info.iterables 		= [('subject_id', subject_list), ('session_id', 
#					  session_list)]


##############################################################################
# 									FUNCTIONS 								 #
##############################################################################

# Function to get subject specific condition information
def dataToTsv(subject_id, session_id):

	from glob import glob
	import numpy as np
	import pandas as pd
	import nipype.interfaces.base as base
	import itertools


	condition_names = ['visual', 'mem_delay', 'nom_delay', 'mem_response', 
					   'nom_response']
	onset_list 		= []
	subjectinfo 	= []
	#print u'extracting subject info'
	files = glob(beh_dir+subject_id+'/S' + session_id +'/'+ u'*.csv')
	for r in range(len(files)):

		be          = pd.read_table(files[r], sep = ';')
		zero        = be.ts_b[0]
		visual 		= list(be.ts_p - zero)
		mem_delay	= list(be.ts_d[be['type'] == 1] - zero)
		nom_delay	= list(be.ts_d[be['type'] == 0] - zero)
		mem_resp	= list(be.ts_r[be['type'] == 1] - zero)
		nom_resp 	= list(be.ts_r[be['type'] == 0] - zero)
		onsets 		= [visual, mem_delay, nom_delay, mem_resp, nom_resp]
		durations = [[1], [8], [8], [3], [3]]

		rep_durations = list(itertools.chain.from_iterable(itertools.repeat(durations[x][0], len(onsets[x])) for x in range(len(durations))))
		rep_conditions = list(itertools.chain.from_iterable(itertools.repeat(condition_names[x], len(onsets[x])) for x in range(len(durations))))
		data = pd.DataFrame({'onset' : sum(onsets, []), 
			'duration' : rep_durations,
			'trial_type' : rep_conditions}) 
		data.to_csv(bids_dir + 'sub-' + subject_id + u'/ses-0' + session_id + '/func/sub-' + subject_id + '_ses-0' + session_id + '_task-wm_run-0' + str(r+1) + '_events.tsv', 
			sep = '\t', index = False)




# read data and output as tsv:
for subj in subject_id:
	dataToTsv(subj, session_id)
