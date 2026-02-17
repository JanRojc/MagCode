import os
import numpy as np
import torch
import traceback

from psbody.mesh import Mesh

from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment
from utils.rotation import normalize_y_rotation
from visualization.blender_renderer import visualize_garment_body

from dataset.canonical_pose_dataset import get_style, get_shape
from visualization.vis_utils import get_specific_pose, get_specific_style_old_tshirt
from visualization.vis_utils import get_specific_shape, get_saved_amass_sequence_thetas, get_any_amass_sequence_thetas
from utils.interpenetration import remove_interpenetration_fast

# Base Output Path
BASE_OUT_PATH = "/mnt/d/ClothSim/Results/TailorNet/"

def get_sequence_inputs(garment_class, gender, amass_sequence, amass_seq_idx):
    """Prepare sequence inputs dynamically for any AMASS sequence."""
    
    # 1. Random/Fixed Shape & Style
    beta = get_specific_shape('somethin')
    if garment_class == 'old-t-shirt':
        gamma = get_specific_style_old_tshirt('big_longsleeve')
    else:
        gamma = get_style('000', gender=gender, garment_class=garment_class)

    # 2. Load AMASS Sequence
    # get_any_amass_sequence_thetas should return (N, 72) pose array and fps
    thetas, mocap_fps = get_any_amass_sequence_thetas(amass_sequence, amass_seq_idx)

    # 3. FPS Downsampling (AMASS is usually 120 or 60, we want 30)
    target_fps = 30  # match ccraft
    subsample_step = int(round(mocap_fps / target_fps)) # usually 120 / 30 = 4
    thetas = thetas[::subsample_step]

    # 4. Tile betas/gammas to match sequence length
    betas = np.tile(beta[None, :], [thetas.shape[0], 1])
    gammas = np.tile(gamma[None, :], [thetas.shape[0], 1])
    
    return thetas, betas, gammas


def process_example(seq_num, seq_idx, garment_class, gender):
    """Runs inference for a single combination and saves PLYs."""
    
    print(f"\n[START] Processing: {seq_num}_{seq_idx} | {gender} | {garment_class}")
    
    # construct output directory matching Maya structure
    # Structure: .../Results/TailorNet/07/01/male/t-shirt/result_ply_files
    out_dir = os.path.join(
        BASE_OUT_PATH,
        str(seq_num),
        str(seq_idx),
        gender,
        garment_class,
        "result_ply_files"
    )

    # 2. Get Inputs
    thetas, betas, gammas = get_sequence_inputs(garment_class, gender, seq_num, f"{seq_num}_{seq_idx}")

    # load model
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
    smpl = SMPL4Garment(gender=gender)

    # make out directory if doesn't exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # run inference
    for i, (theta, beta, gamma) in enumerate(zip(thetas, betas, gammas)):
        # normalize y-rotation to make it front facing
        theta_normalized = normalize_y_rotation(theta)
        with torch.no_grad():
            pred_verts_d = tn_runner.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

        # get garment from predicted displacements
        body, pred_gar = smpl.run(beta=beta, theta=theta_normalized, garment_class=garment_class, garment_d=pred_verts_d)
        pred_gar = remove_interpenetration_fast(pred_gar, body)

        # save body and predicted garment
        body.write_ply(os.path.join(out_dir, "body_{:04d}.ply".format(i)))
        pred_gar.write_ply(os.path.join(out_dir, "pred_gar_{:04d}.ply".format(i)))
        
        if i % 10 == 0:
            print(f"  Frame {i}/{len(thetas)}")

    print(f"[DONE] Saved to {out_dir}")

if __name__ == '__main__':

    sequences = ["01", "02", "05", "07"]
    sequence_indices = ["01", "02", "03", "04", "05"]
    garments = ["t-shirt", "shirt", "pant"] 
    genders = ["male"]
    
    for seq_num in sequences:
        for seq_idx in sequence_indices:
            for gender in genders:
                for garment in garments:
                    try:
                        process_example(seq_num, seq_idx, garment, gender)
                    except Exception as e:
                        print(f"[FAILED] Batch Item {seq_num}_{seq_idx} {garment}: {e}")
                        traceback.print_exc()