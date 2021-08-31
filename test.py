from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir

import os, yaml
from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir

scene_name = 'quad'
scene_video_id = 'video3'
# fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
sdd_root = os.path.join("/home/asyed/OpenTraj/", 'datasets', 'SDD')
annot_file = os.path.join(sdd_root, scene_name, scene_video_id, 'annotations.txt')

# load the homography values
with open(os.path.join(sdd_root, 'estimated_scales.yaml'), 'r') as hf:
    scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
scale = scales_yaml_content[scene_name][scene_video_id]['scale']

traj_dataset = load_sdd(annot_file, scale=scale, scene_id=scene_name + '-' + scene_video_id,
                        drop_lost_frames=False, use_kalman=False)



print(traj_dataset)
path ="/home/asyed/Downloads/VAE-Ped/datasets/stanford/isvc/val/" + str(scene_name) +"_"+ str(scene_video_id[-1:])  +".txt"
df=traj_dataset.data[traj_dataset.data.label == "pedestrian"]
df=df.sort_values(by=["frame_id","agent_id"])
df = df.iloc[:,[0,1,2,3]]
df=df.reset_index()
# df=traj_dataset.data.iloc[:,[0,1,2,3]]
df1 = df.iloc[:,[1,2,3,4]]
print(df)

df1.to_csv(path, header=None, index=None, sep='\t', mode='w')