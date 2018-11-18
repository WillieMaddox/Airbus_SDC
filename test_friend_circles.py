import os
import numpy as np
import pandas as pd
import tqdm
import cv2
from collections import defaultdict
dup_clusters = []
img_id_cluster = []
hash_id_cluster = []


s100 = slice(0, 1)  # 0
s110 = slice(0, 2)  # :2
s111 = slice(0, 3)  # :
s011 = slice(1, 3)  # 1:
s001 = slice(2, 3)  # 2
dim_map = ['100', '110', '111', '011', '001']
dim_map_img_a = {'100': s100, '110': s110, '111': s111, '011': s011, '001': s001}
dim_map_img_b = {'100': s001, '110': s011, '111': s111, '011': s110, '001': s100}


def is_any_b_in_a(a, b):
    # Are any of the 256x256 squares of image A in image B?
    pass


def overlap(img_id, img_cluster):
    if len(img_cluster) == 0:
        return True
    for img_id0 in img_cluster:
        if img_id0 == img_id:
            continue
        for col in dim_map:
            for row in dim_map:
                if np.all(hash_grids[img_id0][dim_map_img_a[col], dim_map_img_a[row]] == hash_grids[img_id][dim_map_img_b[col], dim_map_img_b[row]]):
                    print(col, row)
                    return True


def recurse_hash_to_img(hash_list, image_cluster, hash_cluster):
    for hash_id in hash_list:
        img_list = hash_dict.pop(hash_id, None)
        if img_list is None:
            continue
        hash_cluster.append(hash_id)
        image_cluster, hash_cluster = recurse_img_to_hash(img_list, image_cluster, hash_cluster)
    return image_cluster, hash_cluster


def recurse_img_to_hash(img_list, image_cluster, hash_cluster):
    for img_id in img_list:
        hash_list = img_dict.pop(img_id, None)
        if hash_list is None:
            continue
        if not overlap(img_id, image_cluster):
            continue
        image_cluster.append(img_id)
        image_cluster, hash_cluster = recurse_hash_to_img(hash_list, image_cluster, hash_cluster)
    return image_cluster, hash_cluster


ship_dir = "/media/Borg_LS/DATA/geos/airbus/input/"
image_hash_grids_file = os.path.join(ship_dir, "image_hash_grids.pkl")
df = pd.read_pickle(image_hash_grids_file)

img_dict = {}
hash_grids = {}
for idx, row in df.iterrows():
    hash_grids[row['ImageId']] = row['hash_grid']
    img_dict[row['ImageId']] = row['hash_grid'].flatten()

# img_dict = {d['ImageId']: d['hash_grid'].flatten() for d in hash_dict}
img_ids = list(img_dict)

hash_dict = defaultdict(list)
for img_id, hashes in img_dict.items():
    for h in hashes:
        hash_dict[h].append(img_id)

for img_id in img_ids:
    hash_cluster = []
    image_cluster = []
    hash_list = img_dict.pop(img_id, None)
    if hash_list is None:
        continue
    if not overlap(img_id, image_cluster):
        continue
    image_cluster.append(img_id)
    image_cluster, hash_cluster = recurse_hash_to_img(hash_list, image_cluster, hash_cluster)

    dup_clusters.append({'hash_cluster': hash_cluster, 'image_cluster': image_cluster})


# hash_values = list(hash_dict)

# for hash_value in hash_values:
#     hash_cluster = []
#     image_cluster = []
#     img_list = hash_dict.pop(hash_value, None)
#     if img_list is None:
#         continue
#     hash_cluster.append(hash_value)
#     image_cluster, hash_cluster = recurse_img_to_hash(img_list, image_cluster, hash_cluster)
#
#     dup_clusters.append({'hash_cluster': hash_cluster, 'image_cluster': image_cluster})


print('done')