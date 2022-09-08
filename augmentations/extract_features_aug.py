import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from augmentations import aug_combined

slide_dir = "/media/disk2/prostate/SICAPv2/wsis"
data_root = "/media/disk2/proj_embedding_aug"
extracted_dir = "extracted_mag40x_patch256_fp"
patches_path = os.path.join(data_root, extracted_dir, "patches")

PATCH_SIZE = 256
TARGET_SIZE = 224


filename = "16B0003394"
h5_path = os.path.join(patches_path, filename + "_patches.h5")
with h5py.File(h5_path, "r") as f:
    coords = f["coords"][()]
    x, y = coords[0]

    slide_path = os.path.join(
        slide_dir, filename + ".png"
    )
    slide = Image.open(slide_path)
    slide.load()
    print(slide)
    image = slide.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
    image.load()
    # print(image)

    image.save("images/img_original.png")
    image = np.array(image)

    for i in range(20):
        aug_image = aug_combined(image=image)
        Image.fromarray(aug_image).save(f"images/img_aug{i}.png")


# with os.scandir(patches_path) as it:
#     for entry in it:
#         if not entry.name.startswith(".") and entry.is_file():
#             print(entry.name.replace("_patches.h5", ".png"))
#             with h5py.File(entry.path, "r") as f:
#                 coords = f["coords"][()]
#                 x, y = coords[0]

#                 slide_path = os.path.join(
#                     slide_dir, entry.name.replace("_patches.h5", ".png")
#                 )
#                 slide = Image.open(slide_path)
#                 print(slide)
#                 image = slide.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
#                 image.save("tmp.png")

#                 image = np.array(image)
#                 aug_image = aug_combined(image=image)
#                 Image.fromarray(aug_image).save("tmp_aug.png")

#         break  # only do for first file
