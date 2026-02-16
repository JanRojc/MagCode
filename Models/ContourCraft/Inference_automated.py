from utils.validation import apply_material_params
from utils.validation import load_runner_from_checkpoint
from utils.arguments import load_params
from utils.common import move2device
from utils.io import pickle_dump
from utils.defaults import DEFAULTS
from pathlib import Path

import time
import subprocess


# Set material paramenters, see configs/contourcraft.yaml for the training ranges for each parameter
material_dict = dict()
material_dict['density'] = 0.20022
material_dict['lame_mu'] = 23600.0
material_dict['lame_lambda'] = 44400
material_dict['bending_coeff'] = 3.962e-05


# ====================================================================================================

models_dir = Path(DEFAULTS.data_root) / 'trained_models'

# Choose the model and the configuration file

config_name = 'hood_cvpr'
checkpoint_path = models_dir / 'hood_cvpr.pth'

# config_name = 'hood_final'
# checkpoint_path = models_dir / 'hood_final.pth'

# config_name = 'contourcraft'
# checkpoint_path = models_dir / 'contourcraft.pth'

isHood = "hood" in config_name
if isHood:
    DEFAULTS.results_dir = '/mnt/d/ClothSim/Results/hood/'


# ====================================================================================================


# load the config from a .yaml file and load .py modules specified there
modules, experiment_config = load_params(config_name)

# modify the config to use it for validation 
experiment_config = apply_material_params(experiment_config, material_dict)

# load a Runner object and the .py module it is declared in
runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)


############################################################################################################################################


# file with the pose sequence
from utils.validation import create_postcvpr_one_sequence_dataloader

# If True, the SMPL(-X) poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG 
separate_arms = True

# SMPL
# sequence_path =  Path(DEFAULTS.CMU_root) / '01/01_01_poses.npz'
# sequence_loader = 'cmu_npz_smpl'
# garment_dicts_dir = Path(DEFAULTS.aux_data) / 'garment_dicts' / 'smpl' 
# garment_name = 'hooded_tight_dress'
# gender = 'female'

# SMPL-X
# sequence_path =  Path(DEFAULTS.CMU_root) / '08/08_05_poses.npz'
# sequence_path = Path('examples') / 'fromanypose' / 'mesh_sequence.pkl'
# sequence_loader = 'cmu_npz_smplx'
# garment_dicts_dir = Path(DEFAULTS.aux_data) / 'garment_dicts' / 'smplx'
# garment_name = "celina_002_combined"
# gender = 'female'


def process_example(sequence_num="05", sequence_idx="02", garment="t-shirt", gender='female'):
    sequence_loader = 'cmu_npz_smpl'
    
    sequence_path =  Path(DEFAULTS.CMU_root) / f'{sequence_num}/{sequence_num}_{sequence_idx}_poses.npz'
    if not Path.exists(sequence_path):
        return

    # ccraft garment
    garment_dicts_dir = Path(DEFAULTS.aux_data) / 'garment_dicts' / 'smpl'
    # garment_name = "aaron_009__top"
    # gender = 'male'

    garment_name = f"tailornet_{garment}_{gender}_{sequence_num}"
    # if not Path.exists(garment_dicts_dir / (garment_name+".pkl")):
    #     garment_name = f"tailornet_{garment}_{gender}"
    
    out_path = Path(DEFAULTS.results_dir) / f'output_{sequence_num}_{sequence_idx}_{garment}_{gender}.pkl'
    if Path.exists(out_path):
        print("[ERROR]: output file already exists. skipping:", out_path)

    dataloader = create_postcvpr_one_sequence_dataloader(sequence_path, garment_name, sequence_loader=sequence_loader, 
                                                obstacle_dict_file=None, gender=gender, garment_dicts_dir=garment_dicts_dir)


    ############################################################################################################################################


    start_t = time.time()
    sequence = next(iter(dataloader))
    sequence = move2device(sequence, 'cuda:0')
    trajectories_dict = runner.valid_rollout(sequence,  bare=True, n_steps=500)
    end_t = time.time()


    ############################################################################################################################################


    # Save the sequence to disk
    print(f"Rollout saved into {out_path}")
    pickle_dump(dict(trajectories_dict), out_path)

    seq_len = sequence['cloth'].lookup.shape[1]
    print("Sequence length:", seq_len)
    with open(Path(DEFAULTS.results_dir) / "times.txt", "a", encoding="utf-8") as f:
        f.write(f"{garment} {gender} {sequence_num} {sequence_idx} {(end_t-start_t)/seq_len} sec/it\n")


    command = [
        "cmd.exe", "/c",
        r"C:\Users\janr\Documents\MagCode\.venv_py310\Scripts\python.exe",
        r"C:\Users\janr\Documents\MagCode\Models\ContourCraft\render_automated.py",
        str(isHood)
    ]

    subprocess.run(command, check=True)




sequences = ["01", "02", "05", "07"]
sequence_indices = ["01", "02", "03", "04", "05"]
garments  = ["t-shirt", "shirt", "pant"]

for seq_num in sequences:
    for seq_idx in sequence_indices:
        for garment in garments:
            try:
                process_example(seq_num, seq_idx, garment, "female")
            except Exception as e:
                print(f"[FAILED] {seq_num}_{seq_idx} {garment} female: {e}")