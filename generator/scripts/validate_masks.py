from glob import glob
from PIL import Image
import numpy as np
import os
import numpy as np
import os.path as op
from tqdm import tqdm
import sys

sys.path = ["../code"] + sys.path


def process_mask(seq_name, flag):
    mask_ps = sorted(glob(f"./data/{seq_name}/processed/sam/{flag}/images_masks/*.png"))
    print(f"Processing {seq_name} {flag} with {len(mask_ps)} masks")
    for mask_p in mask_ps:
        mask = Image.open(mask_p)
        mask_np = np.array(mask)
        mask_np[mask_np > 0] = 1
        out_mask = mask_np
        out_mask = out_mask.astype(np.uint8) * 255
        out_p = mask_p.replace("/images_masks/", "/masks_processed/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Image.fromarray(out_mask).save(out_p)

def replace_new_with_original(folder_path):
    new_files = glob(os.path.join(folder_path, "*_new.png"))
    
    for new_file in new_files:
        base_name = new_file.replace("_new.png", ".png")
        if os.path.exists(base_name):
            os.remove(base_name)  # 删除旧文件
        print(f"Renaming: {os.path.abspath(new_file)} to {os.path.abspath(base_name)}")
        os.rename(new_file, base_name)
    
    return sorted(glob(os.path.join(folder_path, "*.png")))

def validate_mask(seq_name):
    print(f"Processing {seq_name}")

    # Step 1: Prepare file paths and load bounding boxes
    rgb_ps = sorted(glob(f"./data/{seq_name}/images/*"))

    # Step 1: format masks

    right_mask_ps = replace_new_with_original(f"./data/{seq_name}/processed/sam/right/images_masks")
    left_mask_ps = replace_new_with_original(f"./data/{seq_name}/processed/sam/left/images_masks")
    object_mask_ps = replace_new_with_original(f"./data/{seq_name}/processed/sam/object/images_masks")

    process_mask(seq_name, "right")
    process_mask(seq_name, "left")
    process_mask(seq_name, "object")

    if len(left_mask_ps) > 0:
        assert len(left_mask_ps) == len(object_mask_ps)

    if len(right_mask_ps) > 0:
        assert len(right_mask_ps) == len(object_mask_ps)

    # rgb image with only object pixels
    rgb_ps = sorted(glob(f"./data/{seq_name}/images/*"))
    object_mask_ps = sorted(
        glob(f"./data/{seq_name}/processed/sam/object/masks_processed/*.png")
    )
    assert len(rgb_ps) == len(object_mask_ps)
    for rgb_p, object_mask_p in zip(rgb_ps, object_mask_ps):
        rgb_np = np.array(Image.open(rgb_p))
        object_mask_np = np.array(Image.open(object_mask_p))
        rgb_np[object_mask_np == 0] = 255

        out_p = rgb_p.replace("/images/", "/processed/images_object/")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Image.fromarray(rgb_np).save(out_p)

    # merge the three masks
    object_mask_ps = sorted(
        glob(f"./data/{seq_name}/processed/sam/object/masks_processed/*.png")
    )
    for object_p in object_mask_ps:
        from src.utils.const import SEGM_IDS

        object_mask = np.array(Image.open(object_p))

        right_p = object_p.replace("/object/", "/right/")
        left_p = object_p.replace("/object/", "/left/")

        out_mask = np.zeros_like(object_mask)

        if op.exists(right_p):
            right_mask = np.array(Image.open(right_p))
            out_mask[right_mask > 0] = SEGM_IDS["right"]
        if op.exists(left_p):
            left_mask = np.array(Image.open(left_p))
            out_mask[left_mask > 0] = SEGM_IDS["left"]
        # object mask overwrites hands
        out_mask[object_mask > 0] = SEGM_IDS["object"]

        out_p = object_p.replace(
            "/processed/sam/object/masks_processed/", "/processed/masks/"
        )
        os.makedirs(os.path.dirname(out_p), exist_ok=True)

        Image.fromarray(out_mask).save(out_p)
    print("Done!")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    validate_mask(args.seq_name)
