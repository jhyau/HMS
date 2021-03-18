import pickle as pkl
import os
import numpy as np
from tempfile import TemporaryFile
from glob import glob
from pathlib import Path

PICKLE_PATH = ['data/containers/shelves/normal/placements/'] #"data/containers/fridges/12252/placements/"]
# Load the output files with np.load(file)
# More info here: https://numpy.org/doc/stable/reference/generated/numpy.save.html

# Output directory
if not os.path.isdir('./imgs_3d/'):
  os.mkdir('./imgs_3d/')

path = './imgs_3d'

# example: data/containers/fridges/12252/placements/shelf_0/Fruit/shelf_setup_4.pkl
for placement in PICKLE_PATH:
  shelves = os.listdir(placement)
  for shelf in shelves:
    objects = os.listdir(os.path.join(placement, shelf))
    for obj in objects:
      #path = os.path.join(os.path.join(placement, shelf), obj)
      files = glob(os.path.join(os.path.join(placement, shelf), obj) + '/*.pkl')
      for file in files:
        out_path = os.path.join(os.path.join(os.path.join('./imgs_3d/', shelf), obj), file.split('/')[-1].split('.')[0])
        with open(file, 'rb') as out:
          p = pkl.load(out)
          keys = p.keys()
          for key in keys:
            path = os.path.join(out_path, str(key))
            #print(f'{p[key].keys()}')
            if 'im3d' not in p[key].keys():
              continue
            # outfile = TemporaryFile()
            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_im3d'
            print(f'{title}')
            print(f'{path}')
            Path(path).mkdir(parents=True, exist_ok=True)
            np.save(path+'/'+title, p[key]['im3d'])
            np.save(path+'/'+title+'_rand', p[key]['im3d_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_im3d'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_im3d'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_im3d'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_im3d'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_im3d'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_im3d'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_im3d'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_depth_im3d'
            np.save(path+'/'+title, p[key]['depth_im3d'])
            np.save(path+'/'+title+'_rand', p[key]['depth_im3d_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_depth_im3d'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_depth_im3d'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_depth_im3d'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_depth_im3d'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_depth_im3d'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_depth_im3d'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_depth_im3d'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_location'
            np.save(path+'/'+title, p[key]['location'])
            np.save(path+'/'+title, p[key][str(key)+'_location'])
            np.save(path+'/'+title+'_location_rand', p[key]['location_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_location'])
            np.save(path+'/'+title+'_x_add_mean_location', p[key]['x_add_mean_location'])
            np.save(path+'/'+title+'_x_sub_mean_location', p[key]['x_sub_mean_location'])
            np.save(path+'/'+title+'_y_add_mean_location', p[key]['y_add_mean_location'])
            np.save(path+'/'+title+'_y_sub_mean_location', p[key]['y_sub_mean_location'])
            np.save(path+'/'+title+'_z_add_mean_location', p[key]['z_add_mean_location'])
            np.save(path+'/'+title+'_z_sub_mean_location', p[key]['z_sub_mean_location'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_segmask'
            np.save(path+'/'+title, p[key][str(key)+'_segmask'])
            np.save(path+'/'+title+'_rand', p[key][str(key)+'_segmask_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_'+str(key)+'_segmask'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_'+str(key)+'_segmask'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_'+str(key)+'_segmask'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_'+str(key)+'_segmask'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_'+str(key)+'_segmask'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_'+str(key)+'_segmask'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_'+str(key)+'_segmask'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_pose_trans'
            np.save(path+'/'+title, p[key][str(key)+'_pose_trans'])
            np.save(path+'/'+title+'_rand', p[key][str(key)+'_pose_trans_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_'+str(key)+'_pose_trans'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_'+str(key)+'_pose_trans'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_'+str(key)+'_pose_trans'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_'+str(key)+'_pose_trans'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_'+str(key)+'_pose_trans'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_'+str(key)+'_pose_trans'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_'+str(key)+'_pose_trans'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_pose_rot'
            np.save(path+'/'+title, p[key][str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_rand', p[key][str(key)+'_pose_rot_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_'+str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_'+str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_'+str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_'+str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_'+str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_'+str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_'+str(key)+'_pose_rot'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_pose'
            np.save(path+'/'+title, p[key][str(key)+'_pose'])
            np.save(path+'/'+title+'_rand', p[key][str(key)+'_pose_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_'+str(key)+'_pose_rot'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_'+str(key)+'_pose'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_'+str(key)+'_pose'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_'+str(key)+'_pose'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_'+str(key)+'_pose'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_'+str(key)+'_pose'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_'+str(key)+'_pose'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_mat'
            np.save(path+'/'+title, p[key][str(key)+'_mat'])
            np.save(path+'/'+title+'_rand', p[key][str(key)+'_mat_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_'+str(key)+'_mat_rot'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_'+str(key)+'_mat'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_'+str(key)+'_mat'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_'+str(key)+'_mat'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_'+str(key)+'_mat'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_'+str(key)+'_mat'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_'+str(key)+'_mat'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_camera_intrinsics'
            np.save(path+'/'+title, p[key]['K'])
            np.save(path+'/'+title+'_rand', p[key]['K_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_K'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_K'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_K'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_K'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_K'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_K'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_K'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_camera'
            np.save(path+'/'+title, p[key]['camera'])
            np.save(path+'/'+title+'_rand', p[key]['camera_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_K'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_camera'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_camera'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_camera'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_camera'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_camera'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_camera'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_target'
            np.save(path+'/'+title, p[key]['target'])
            np.save(path+'/'+title+'_rand', p[key]['target_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_target'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_target'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_target'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_target'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_target'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_target'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_target'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_up'
            np.save(path+'/'+title, p[key]['up'])
            np.save(path+'/'+title+'_rand', p[key]['up_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_up'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_up'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_up'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_up'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_up'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_up'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_up'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_P'
            np.save(path+'/'+title, p[key]['P'])
            np.save(path+'/'+title+'_rand', p[key]['P_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_P'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_P'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_P'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_P'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_P'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_P'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_P'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_V'
            np.save(path+'/'+title, p[key]['V'])
            np.save(path+'/'+title+'_rand', p[key]['V_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_V'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_V'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_V'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_V'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_V'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_V'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_V'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_lightP'
            np.save(path+'/'+title, p[key]['lightP'])
            np.save(path+'/'+title+'_rand', p[key]['lightP_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_lightP'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_lightP'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_lightP'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_lightP'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_lightP'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_lightP'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_lightP'])

            title = shelf + '_' + obj + '_' + file.split('/')[-1].split('.')[0] + '_' + str(key) + '_lightV'
            np.save(path+'/'+title, p[key]['lightV'])
            np.save(path+'/'+title+'_rand', p[key]['lightV_rand'])
            np.save(path+'/'+title+'_mean_loc', p[key]['mean_loc_lightV'])
            np.save(path+'/'+title+'_x_add_mean_loc', p[key]['x_add_mean_loc_lightV'])
            np.save(path+'/'+title+'_x_sub_mean_loc', p[key]['x_sub_mean_loc_lightV'])
            np.save(path+'/'+title+'_y_add_mean_loc', p[key]['y_add_mean_loc_lightV'])
            np.save(path+'/'+title+'_y_sub_mean_loc', p[key]['y_sub_mean_loc_lightV'])
            np.save(path+'/'+title+'_z_add_mean_loc', p[key]['z_add_mean_loc_lightV'])
            np.save(path+'/'+title+'_z_sub_mean_loc', p[key]['z_sub_mean_loc_lightV'])

