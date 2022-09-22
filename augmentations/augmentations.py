import albumentations as A
import cv2
import random


def aug_combined(image):
    """Perform random augmentation on image"""

    color_jitter = A.HueSaturationValue(
        hue_shift_limit=(-20, +20),
        sat_shift_limit=(-30, +60),
        val_shift_limit=(-20, +20),
        always_apply=True,
    )

    random_angle = [0, 90, 180, 270][random.randint(0, 3)]

    # TODO: zoom out with context of WSI
    rotation_zoom = A.ShiftScaleRotate(
        shift_limit=0,
        scale_limit=(-0.1, +0.1),
        rotate_limit=(random_angle,) * 2,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        rotate_method="largest_box",
        always_apply=True,
    )

    transform = A.Compose(
        [
            color_jitter,
            rotation_zoom,
        ]
    )
    return transform(image=image)["image"]


def aug_rotation(image):
    """Augment image with rotation (int from 0 to 360)"""

    transform = A.Rotate(
        (90,) * 2,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        method="largest_box",
        always_apply=True,
    )
    return transform(image=image)["image"]

def aug_zoom(image):
    """Augment image with zoom percentage (int from -0.1 to +0.1)"""

    transform = A.ShiftScaleRotate(
        shift_limit=0,
        scale_limit=(-0.1, +0.1),
        rotate_limit=0,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        always_apply=True,
    )
    return transform(image=image)["image"]


def aug_hue(image):
    """Augment image with hue (int from -100 to 100)"""

    transform = A.HueSaturationValue(
        hue_shift_limit=(-20, +20),
        sat_shift_limit=0,
        val_shift_limit=0,
        always_apply=True,
    )
    return transform(image=image)["image"]


def aug_saturation(image):
    """Augment image with sat (int from -100 to 100)"""

    transform = A.HueSaturationValue(
        hue_shift_limit=0,
        sat_shift_limit=(-30, +60),
        val_shift_limit=0,
        always_apply=True,
    )
    return transform(image=image)["image"]


def aug_value(image):
    """Augment image with val (int from -100 to 100)"""

    transform = A.HueSaturationValue(
        hue_shift_limit=0,
        sat_shift_limit=0,
        val_shift_limit=(-20, +20),
        always_apply=True,
    )
    return transform(image=image)["image"]
