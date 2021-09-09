import nibabel as nib
import glob as glob
from nilearn.plotting import plot_stat_map
from nilearn import datasets, surface, plotting

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

subject = 'C06' # 'C', 'E', 'S', 'all'
session = '1'

base_dir = '/storage/albamrt/NMDA/MRI/out/2nd_level/'
contrasts = ['visual', 'delay', 'response', 'memory', 'recall']

for contrast in contrasts:
	print(contrast)

	subjects = os.listdir(base_dir + contrast)  

	for subject in subjects:

		print(subject)
		# get statistical map:
		stat_img_files = glob.glob(base_dir + contrast + '/' + subject + '/ses-0' + session + '/in_file/zstat*.nii.gz')

		for stat_img_file in stat_img_files: 
			
			stat_img = nib.load(stat_img_file)
			filename = os.path.basename(stat_img_file)[0:-7]
			#all_sub_img = nib.load(glob.glob(base_dir + contrast + '/grp_all/sub-all/ses-0' + session + '/func/sub-all_ses-0' + session + '*fwe_zstat.nii.gz')[0])


			glass_img = plotting.plot_glass_brain(stat_img,
			                          title = subject + ' S' + session + ' ' + contrast,
			                          display_mode = 'lyrz',
			                          plot_abs = False,
			                          colorbar = True,
			                          vmin = -6,
			                          vmax = 6
			                         )
			glass_img.savefig(base_dir + contrast + '/' + subject + '/ses-0' + session + '/in_file/glass_' + filename + '_clust.png')
			del glass_img


			glass_img_html = plotting.view_img(stat_img, 
												symmetric_cmap = True, 
												vmax = 6, 
												cut_coords = [-42, -16, 52], 
												title = subject + ' S' + session + ' ' + contrast )   
			glass_img_html.save_as_html(base_dir + contrast + '/' + subject + '/ses-0' + session + '/in_file/glass_' + filename + '_clust_viewer.html')
			del glass_img_html
			

			# Plotting statistical maps on brain sections (group level):
			sections = ['z', 'x', 'y']
			for section in sections:
			    sections_img = plotting.plot_stat_map(stat_img, 
			                  threshold = 0,
			                  display_mode = section, 
			                  cut_coords = 4, 
			                  black_bg = False,
			                  vmax = 6,
			                  title = subject + ' S' + session + ' ' + contrast)
			    sections_img.savefig(base_dir + contrast + '/' + subject + '/ses-0' + session + '/in_file/sections_' + filename + '_clust_' + section + '.png')
			    del sections_img

