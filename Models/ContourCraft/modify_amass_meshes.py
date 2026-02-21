import os
import time
import numpy as np
import torch
from pathlib import Path

# CCraft / HOOD imports
from utils.defaults import DEFAULTS
from utils.validation import create_postcvpr_one_sequence_dataloader

# ======================================================================================
# CONFIGURATION
# ======================================================================================
ORIGINAL_AMASS_ROOT = Path(DEFAULTS.CMU_root)
MODIFIED_AMASS_ROOT = Path("/mnt/d/ClothSim/AMASS_Modified/CMU/") # Update this if needed

# Using the focused subset
SEQUENCES = ["01"]
SEQUENCE_INDICES = ["01"]
GENDER = "male"
DUMMY_GARMENT = "t-shirt" 

def process_and_save_sequence(seq_num, seq_idx, gender):
    seq_name = f"{seq_num}_{seq_idx}"
    original_npz_path = ORIGINAL_AMASS_ROOT / seq_num / f"{seq_name}_poses.npz"
    
    if not original_npz_path.exists():
        print(f"[-] Skipping {seq_name} (Original file not found)")
        return

    out_dir = MODIFIED_AMASS_ROOT / seq_num
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz_path = out_dir / f"{seq_name}_poses.npz"

    print(f"[+] Extracting modified pose for {seq_name}...")
    
    garment_dicts_dir = Path(DEFAULTS.aux_data) / 'garment_dicts' / 'smpl'
    garment_name = f"tailornet_{DUMMY_GARMENT}_{gender}_{seq_num}"
    if not (garment_dicts_dir / f"{garment_name}.pkl").exists():
        garment_name = f"tailornet_{DUMMY_GARMENT}_{gender}"

    try:
        dataloader = create_postcvpr_one_sequence_dataloader(
            sequence_path=original_npz_path, 
            garment_name=garment_name, 
            sequence_loader='cmu_npz_smpl', 
            obstacle_dict_file=None, 
            gender=gender, 
            garment_dicts_dir=garment_dicts_dir
        )
    except Exception as e:
        print(f" [!] Dataloader failed for {seq_name}: {e}")
        return

    # 1. Dig down to the sequence loader
    dataset = dataloader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    seq_loader = dataset.loader.sequence_loader

    # 2. Actively call the loader with the absolute path
    # (os.path.join safely returns the absolute path if fname is already absolute)
    try:
        processed_result = seq_loader.load_sequence(str(original_npz_path))
    except Exception as e:
        print(f" [!] Failed calling load_sequence(): {e}")
        return

    # 3. Stitch the arrays back together!
    if not isinstance(processed_result, dict) or 'body_pose' not in processed_result:
        print(" [!] Missing expected keys in processed result.")
        return

    global_orient = processed_result['global_orient'] # Shape: (N, 3)
    body_pose = processed_result['body_pose']         # Shape: (N, 69)
    
    mod_pose = np.concatenate([global_orient, body_pose], axis=1) # Shape: (N, 72)
    mod_trans = processed_result['transl']
    mod_betas = processed_result['betas']

    # Strip PyTorch tensors if present
    if hasattr(mod_pose, 'detach'): mod_pose = mod_pose.detach().cpu().numpy()
    if hasattr(mod_trans, 'detach'): mod_trans = mod_trans.detach().cpu().numpy()
    if hasattr(mod_betas, 'detach'): mod_betas = mod_betas.detach().cpu().numpy()

    # 4. Save a fresh, clean .npz file exclusively for TailorNet/Maya
    # We set mocap_framerate to the downsampled rate (usually 30) so 
    # TailorNet doesn't accidentally downsample it again.
    target_fps = seq_loader.mcfg.fps if hasattr(seq_loader.mcfg, 'fps') else 30

    new_data = {
        'poses': mod_pose,
        'trans': mod_trans,
        'betas': mod_betas,
        'gender': str(gender),
        'mocap_framerate': np.array(target_fps)
    }

    np.savez_compressed(out_npz_path, **new_data)
    print(f" -> Successfully built and saved clean AMASS file to {out_npz_path}")


if __name__ == "__main__":
    start_time = time.time()
    print("=== Starting AMASS Modified Pose Extraction ===")
    
    for seq_num in SEQUENCES:
        for seq_idx in SEQUENCE_INDICES:
            process_and_save_sequence(seq_num, seq_idx, GENDER)
            
    print(f"=== Finished in {time.time() - start_time:.1f} seconds ===")