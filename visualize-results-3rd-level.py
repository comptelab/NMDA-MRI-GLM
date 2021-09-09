import nibabel as nib
import glob as glob
from nilearn.plotting import plot_stat_map
from nilearn import datasets, surface, plotting

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

group = 'C' # 'C', 'E', 'S', 'all'
session = '1'

base_dir = '/storage/albamrt/NMDA/MRI/out/3rd_level/'
contrasts = ['visual', 'delay', 'response', 'memory', 'recall']

for contrast in contrasts:
	print(contrast)

	# get statistical map:
	stat_img = nib.load(glob.glob(base_dir + group + '_' + contrast + '/grp_all/sub-all/ses-0' + session + '/func/sub-all_ses-0' + session + '*clust_zstat.nii.gz')[0])
	all_sub_img = nib.load(glob.glob(base_dir + group + '_' + contrast + '/grp_all/sub-all/ses-0' + session + '/func/sub-all_ses-0' + session + '*fwe_zstat.nii.gz')[0])


	glass_img = plotting.plot_glass_brain(stat_img,
	                          title = group + ' S' + session + ' ' + contrast,
	                          display_mode = 'lyrz',
	                          plot_abs = False,
	                          colorbar = True,
	                          vmin = -6,
	                          vmax = 6
	                         )
	glass_img.savefig(base_dir + group + '_' + contrast + '/grp_all/sub-all/ses-0' + session + '/func/' + contrast + '_glass_zstat_clust.png')


	glass_img_html = plotting.view_img(stat_img, 
										symmetric_cmap = True, 
										vmax = 6, 
										cut_coords = [-42, -16, 52], 
										title = group + ' S' + session + ' ' + contrast )   
	glass_img_html.save_as_html(base_dir + group + '_' + contrast + '/grp_all/sub-all/ses-0' + session + '/func/' + contrast +'_glass_zstat_clust_viewer.html')
	

	# Plotting statistical maps on brain sections (group level):
	sections = ['z', 'x', 'y']
	for section in sections:
	    sections_img = plotting.plot_stat_map(stat_img, 
	                  threshold = 0,
	                  display_mode = section, 
	                  cut_coords = 4, 
	                  black_bg = False,
	                  vmax = 6,
	                  title = group + ' S' + session + ' ' + contrast)
	    sections_img.savefig(base_dir + group + '_' + contrast + '/grp_all/sub-all/ses-0' + session + '/func/' + contrast + '_sections_zstat_clust_' + section + '.png')

