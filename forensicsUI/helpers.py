from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import ast


def convert_bbox(bboxes):
    output = []
    for bbox in bboxes.split("-"):
        arr_data = ast.literal_eval(bbox)
        output.append(arr_data)
    return output


def convert_list_to_string(arr):
    return "-".join([str(x) for x in arr])


def convert_string_to_list(arr_str):
    return [int(x) for x in arr_str.split("-")]


# TODO implement better logic for picking target object
def save_target_obj(results):
    #      x          y             x           y         conf      cls
    # [6.0955e+01, 2.4356e+02, 2.8908e+02, 3.8819e+02, 9.3720e-01, 2.0000e+00]
    max_conf = 0
    max_conf_res = []
    for result in results[0].boxes.data:
        if result[4] > max_conf:
            max_conf_res = result
            max_conf = result[4]

    image_result = max_conf_res
    img = Image.fromarray(results[0].orig_img)
    x, y, m, n = image_result[:4].to(int).tolist()
    cropped_image = img.crop((x, y, m, n))
    cropped_image.save("target_obj.jpg")
    return image_result


# todo maybe remove
def from_bytes(bytes_data, shape):
    return np.frombuffer(bytes_data, dtype=np.float32).reshape(shape)


# todo remove
"""def to_int(arr):
    """"""
    return [int(x) for x in arr]
"""


def plot_imgs(labels, im_arrs):
    """visualize several images given their labels"""
    # Arrange the images in a grid
    rows = 2
    cols = 3  # Ensure cols can accommodate the number of images

    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))

    # Plot each image on its corresponding axis, adding labels
    for i, (image, label, ax) in enumerate(zip(im_arrs, labels, axes.flatten())):
        ax.imshow(image, cmap='gray')  # Use 'gray' colorscale for grayscale images
        ax.set_title(label)
        ax.axis('off')  # Remove axis ticks and labels

    # Adjust spacing if needed
    plt.tight_layout()
    plt.show()


def crop_object(bbox, frame):
    """crop object from frame given bbox"""
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


# cls, feature_shape, frame_shape, img_path, input_video_path, num_of_objects_detected,
def write_metadata_to_file(metadata):
    """Metadata handling"""
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)


def reshape_target(arr):
    return arr.squeeze()


def reshape_feature(arr):
    """remove unnecessary dimensions"""
    return arr.reshape(1, -1)
