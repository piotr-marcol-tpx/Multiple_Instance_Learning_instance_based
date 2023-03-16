import albumentations as A
from torchvision import transforms

prob = 0.75
p_soft = 0.5

# DATA NORMALIZATION
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pipeline_transform_paper = A.Compose([
    # A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
    # A.RandomCrop(height=220, width=220, p=prob),
    # A.Resize(224,224,always_apply=True),
    # A.MedianBlur(blur_limit=3, p=prob),
    # A.CropAndPad(percent=(-0.01, -0.05),pad_mode=1,always_apply=True),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), always_apply=True),
    A.VerticalFlip(p=prob),
    A.HorizontalFlip(p=prob),
    A.RandomRotate90(p=prob),
    # A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),always_apply=True),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, always_apply=True),
    A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, always_apply=True),
    # A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
    # A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
    # A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=prob),
    # A.RandomBrightness(limit=0.2, p=prob),
    # A.RandomContrast(limit=0.2, p=prob),
    # A.GaussNoise(p=prob),
    # A.ElasticTransform(alpha=2,border_mode=4, sigma=20, alpha_affine=20, p=prob, always_apply=True),
    # A.GridDistortion(num_steps=2, distort_limit=0.2, interpolation=1, border_mode=4, p=prob),
    # A.GlassBlur(sigma=0.3, max_delta=2, iterations=1, p=prob),
    # A.OpticalDistortion (distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob),
    # A.GridDropout (ratio=0.3, unit_size_min=3, unit_size_max=40, holes_number_x=3, holes_number_y=3, shift_x=1, shift_y=10, random_offset=True, fill_value=0, p=prob),
    # A.Equalize(p=prob),
    # A.Posterize(p=prob, always_apply=True),
    # A.RandomGamma(p=prob, always_apply=True),
    # A.Superpixels(p_replace=0.05, n_segments=100, max_size=128, interpolation=1, p=prob),
    # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=prob),
    A.ToGray(p=0.2),
    # A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=0, p=prob),
    # A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=255, p=prob),
    ])

pipeline_transform = A.Compose([
    # A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
    # A.MedianBlur(blur_limit=3, p=prob),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), always_apply=True),
    A.VerticalFlip(p=prob),
    A.HorizontalFlip(p=prob),
    A.RandomRotate90(p=prob),
    # A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),always_apply=True),
    A.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.6, hue=0.2, always_apply=True),
    A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, always_apply=True),
    # A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
    # A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=prob),
    # A.RandomBrightness(limit=0.2, p=prob),
    # A.RandomContrast(limit=0.2, p=prob),
    # A.GaussNoise(p=prob),
    # A.ElasticTransform(alpha=2, border_mode=4, sigma=10, alpha_affine=10, p=prob),
    # A.GridDistortion(num_steps=1, distort_limit=0.1, interpolation=1, border_mode=4, p=prob),
    # A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob),
    A.JpegCompression(quality_lower=95, p=prob),
    # A.Equalize(p=prob),
    # A.Posterize(p=prob, always_apply=True),
    # A.RandomGamma(p=prob, always_apply=True),
    A.ToGray(p=0.2)
    ])


pipeline_transform_soft = A.Compose([
    # A.ElasticTransform(alpha=0.01,p=p_soft),
    # A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=p_soft),
    A.HueSaturationValue(hue_shift_limit=(-15, 8), sat_shift_limit=(-20, 10), val_shift_limit=(-8, 8), p=p_soft),
    A.VerticalFlip(p=p_soft),
    A.HorizontalFlip(p=p_soft),
    A.RandomRotate90(p=p_soft),
    # A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),p=p_soft),
    # A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=p_soft),
    # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=p_soft),
    # A.RandomBrightness(limit=0.1, p=p_soft),
    # A.RandomContrast(limit=0.1, p=p_soft),
    ])
