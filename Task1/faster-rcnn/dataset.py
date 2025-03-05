import os
from PIL import Image
from collections import defaultdict
import pycocotools
from detectron2.structures import BoxMode


def build_kitti_mots_dicts(dataset_path, instances_ids=None):
    # Check that the dataset path exists and is a directory.
    assert os.path.isdir(dataset_path), f"Dataset path '{dataset_path}' is not a directory or does not exist."

    ann_dir = os.path.join(dataset_path, "instances_txt")
    images_folder = os.path.join(dataset_path, "training", "image_02")

    # Check that the annotation directory exists.
    assert os.path.isdir(ann_dir), f"Annotation directory '{ann_dir}' does not exist or is not a directory."
    # Check that the images folder exists.
    assert os.path.isdir(images_folder), f"Images folder '{images_folder}' does not exist or is not a directory."

    dataset_dicts = []
    category_map = {1: 1, 2: 0}  # car -> 1, pedestrian -> 0

    # Iterate over each instance directory in the images folder.
    for instance_dir in sorted(os.listdir(images_folder)):
        instance_dir_path = os.path.join(images_folder, instance_dir)
        if not os.path.isdir(instance_dir_path):
            continue  # Skip non-directory entries

        try:
            instance_number = int(instance_dir)
        except ValueError:
            continue

        if instances_ids and instance_number not in instances_ids:
            continue

        instance_path = instance_dir_path
        ann_file = os.path.join(ann_dir, f"{instance_number:04d}.txt")
        
        annotations = defaultdict(list)
        if os.path.exists(ann_file):
            # Make sure ann_file is a file.
            assert os.path.isfile(ann_file), f"Annotation file '{ann_file}' exists but is not a file."
            with open(ann_file, "r") as file:
                for line in file:
                    parts = line.strip().split(" ")
                    # Validate that the annotation line has at least 6 parts.
                    if len(parts) < 6:
                        raise ValueError(f"Annotation line has less than 6 parts: {line}")
                    try:
                        time_frame = int(parts[0])
                        class_id = int(parts[2])
                        img_height = int(parts[3])
                        img_width = int(parts[4])
                    except ValueError:
                        raise ValueError(f"Annotation line contains non-integer values where expected: {line}")

                    # Validate that image dimensions are positive.
                    assert img_height > 0 and img_width > 0, f"Invalid image dimensions in annotation: {img_height}x{img_width}"

                    rle = parts[5].strip()
                    # Validate that RLE string is not empty.
                    assert rle, f"Empty RLE in annotation line: {line}"

                    # If the class id is 10, skip this annotation.
                    if class_id == 10:
                        continue

                    # Ensure that the class_id is in the category_map.
                    assert class_id in category_map, f"Class ID {class_id} is not in the category_map {category_map}"

                    mask = {"counts": rle, "size": [img_height, img_width]}
                    bbox = pycocotools.mask.toBbox(mask).tolist()
                    
                    """ IMPLEMENT IF SEGMENTATION IS NEEDED
                    # Convert to uint8 image for contour extraction.
                    segmentation = pycocotools.mask.decode(mask)
                    # Check that the segmentation mask dimensions match the expected image dimensions.
                    assert segmentation.shape[0] == img_height and segmentation.shape[1] == img_width, (
                        f"Segmentation shape {segmentation.shape} does not match expected dimensions [{img_height}, {img_width}]"
                    )
                    seg_uint8 = segmentation.astype(np.uint8)
                    contours, hierarchy = cv2.findContours(seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Ensure that at least one contour was found.
                    assert contours is not None and len(contours) > 0, f"No contours found in segmentation for annotation line: {line}"
                    polygons = [contour.flatten().tolist() for contour in contours if len(contour) >= 6]
                    # Ensure that at least one valid polygon exists.
                    assert len(polygons) > 0, f"No valid polygon (at least 3 points) found in annotation line: {line}"
                    """
                    polygons = []
                    
                    # Validate that the bounding box values are non-negative and have positive width/height.
                    x, y, w, h = bbox
                    assert x >= 0 and y >= 0 and w > 0 and h > 0, f"Invalid bbox values: {bbox}"

                    # Create a unique image ID
                    image_id = int(f"{instance_number:04d}{time_frame:06d}")

                    annotations[image_id].append({
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": polygons,
                        "category_id": category_map[class_id],
                    })

        # Check that the instance_path exists and is a directory.
        assert os.path.isdir(instance_path), f"Instance path '{instance_path}' is not a directory."
        image_files = sorted(os.listdir(instance_path))
        # Ensure that the instance directory contains image files.
        assert len(image_files) > 0, f"No image files found in instance directory '{instance_path}'."
        for img_file in image_files:
            # Process only valid image files based on extension.
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue
            try:
                time_frame = int(os.path.splitext(img_file)[0])
            except ValueError:
                raise ValueError(f"Image filename '{img_file}' does not contain a valid integer time frame.")
            image_id = int(f"{instance_number:04d}{time_frame:06d}")
            file_path = os.path.join(instance_path, img_file)
            # Verify that the image file exists.
            assert os.path.isfile(file_path), f"Image file '{file_path}' does not exist or is not a file."
            # Open image and check dimensions.
            with Image.open(file_path) as img:
                width, height = img.size
                # Validate that the image dimensions are positive.
                assert width > 0 and height > 0, f"Image '{file_path}' has invalid dimensions: {width}x{height}"

            record = {
                "file_name": file_path,
                "image_id": image_id,
                "height": height,
                "width": width,
                "annotations": annotations[image_id],
            }
            dataset_dicts.append(record)

    # Final check: ensure that the dataset is not empty.
    assert len(dataset_dicts) > 0, "No data found in dataset, please check the dataset path and structure."
    return dataset_dicts
