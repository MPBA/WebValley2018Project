import tensorlayer as tl
import numpy as np
import os, csv, random, gc, pickle
import nibabel as nib
from skimage import transform

CHANNELS = 1

"""
In seg file
--------------
Label 1: necrotic and non-enhancing tumor
Label 2: edema 
Label 4: enhancing tumor
Label 0: background

MRI
-------
whole/complete tumor: 1 2 4
core: 1 4
enhance: 4
"""
###============================= SETTINGS ===================================###
DATA_SIZE = 'all'  # (small, half or all)

save_dir = "data/train_dev_all/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

HGG_data_path = "BRATS_DATASET/HGG"
LGG_data_path = "BRATS_DATASET/LGG"
survival_csv_path = "BRATS_DATASET/survival_data.csv"

###==========================================================================###

survival_id_list = []
survival_age_list = []
survival_peroid_list = []

with open(survival_csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for idx, content in enumerate(reader):
        survival_id_list.append(content[0])
        survival_age_list.append(float(content[1]))
        survival_peroid_list.append(float(content[2]))

print(len(survival_id_list))  # 163

if DATA_SIZE == 'all':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)
elif DATA_SIZE == 'half':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:100]  # DEBUG WITH SMALL DATA
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:30]  # DEBUG WITH SMALL DATA
elif DATA_SIZE == 'small':
    HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)[0:50]  # DEBUG WITH SMALL DATA
    LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)[0:20]  # DEBUG WITH SMALL DATA
else:
    exit("Unknow DATA_SIZE")

print(len(HGG_path_list), len(LGG_path_list))  # 210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]
LGG_name_list = [os.path.basename(p) for p in LGG_path_list]

survival_id_from_HGG = []
survival_id_from_LGG = []
for i in survival_id_list:
    if i in HGG_name_list:
        survival_id_from_HGG.append(i)
    elif i in LGG_name_list:
        survival_id_from_LGG.append(i)
    else:
        print(i)

print(len(survival_id_from_HGG), len(survival_id_from_LGG))  # 163, 0

# use 42 from 210 (in 163 subset) and 15 from 75 as 0.8/0.2 train/dev split

# use 126/42/42 from 210 (in 163 subset) and 45/15/15 from 75 as 0.6/0.2/0.2 train/dev/test split
index_HGG = list(range(0, len(survival_id_from_HGG)))
index_LGG = list(range(0, len(LGG_name_list)))
# random.shuffle(index_HGG)
# random.shuffle(index_HGG)

if DATA_SIZE == 'all':
    dev_index_HGG = index_HGG[-84:-42]
    test_index_HGG = index_HGG[-42:]
    tr_index_HGG = index_HGG[:-84]
    dev_index_LGG = index_LGG[-30:-15]
    test_index_LGG = index_LGG[-15:]
    tr_index_LGG = index_LGG[:-30]
elif DATA_SIZE == 'half':
    dev_index_HGG = index_HGG[-30:]  # DEBUG WITH SMALL DATA
    test_index_HGG = index_HGG[-5:]
    tr_index_HGG = index_HGG[:-30]
    dev_index_LGG = index_LGG[-10:]  # DEBUG WITH SMALL DATA
    test_index_LGG = index_LGG[-5:]
    tr_index_LGG = index_LGG[:-10]
elif DATA_SIZE == 'small':
    dev_index_HGG = index_HGG[35:42]  # DEBUG WITH SMALL DATA
    # print(index_HGG, dev_index_HGG)
    # exit()
    test_index_HGG = index_HGG[41:42]
    tr_index_HGG = index_HGG[0:35]
    dev_index_LGG = index_LGG[7:10]  # DEBUG WITH SMALL DATA
    test_index_LGG = index_LGG[9:10]
    tr_index_LGG = index_LGG[0:7]

survival_id_dev_HGG = [survival_id_from_HGG[i] for i in dev_index_HGG]
survival_id_test_HGG = [survival_id_from_HGG[i] for i in test_index_HGG]
survival_id_tr_HGG = [survival_id_from_HGG[i] for i in tr_index_HGG]

survival_id_dev_LGG = [LGG_name_list[i] for i in dev_index_LGG]
survival_id_test_LGG = [LGG_name_list[i] for i in test_index_LGG]
survival_id_tr_LGG = [LGG_name_list[i] for i in tr_index_LGG]

survival_age_dev = [survival_age_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_age_test = [survival_age_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_age_tr = [survival_age_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

survival_period_dev = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_dev_HGG]
survival_period_test = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_test_HGG]
survival_period_tr = [survival_peroid_list[survival_id_list.index(i)] for i in survival_id_tr_HGG]

data_types = ['flair', 't1', 't1ce', 't2']
data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}

# ==================== LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD
for i in data_types:
    data_temp_list = []
    for j in HGG_name_list:
        img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)

    for j in LGG_name_list:
        img_path = os.path.join(LGG_data_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data()
        data_temp_list.append(img)

    data_temp_list = np.asarray(data_temp_list)
    m = np.mean(data_temp_list)
    s = np.std(data_temp_list)
    data_types_mean_std_dict[i]['mean'] = m
    data_types_mean_std_dict[i]['std'] = s
del data_temp_list

print(data_types_mean_std_dict)

with open(save_dir + 'mean_std_dict.pickle', 'wb') as f:
    pickle.dump(data_types_mean_std_dict, f, protocol=4)

# ==================== GET NORMALIZE IMAGES

X_train_input = []
y_train_target = []

X_dev_input = []
y_dev_target = []

print(" HGG Validation")
REF_CLASS = 0
for i in survival_id_dev_HGG:
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = transform.resize(img, output_shape=(28, 28, CHANNELS))
        img = img.astype(np.float32)
        print(img.shape)
        X_dev_input.append(img)
        y_dev_target.append(REF_CLASS)
    print("finished {}".format(i))

print(" LGG Validation")
REF_CLASS = 1
for i in survival_id_dev_LGG:
    for j in data_types:
        img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = transform.resize(img, output_shape=(28, 28, CHANNELS))
        img = img.astype(np.float32)
        print(img.shape)
        X_dev_input.append(img)
        y_dev_target.append(REF_CLASS)
    print("finished {}".format(i))

X_dev_input = np.asarray(X_dev_input, dtype=np.float32)
y_dev_target = np.asarray(y_dev_target, dtype=np.uint8)

print(X_dev_input.shape)
print(y_dev_target.shape)

print(" HGG Train")
REF_CLASS = 0
for i in survival_id_tr_HGG:
    for j in data_types:
        img_path = os.path.join(HGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = transform.resize(img, output_shape=(28, 28, CHANNELS))
        img = img.astype(np.float32)
        X_train_input.append(img)
        y_train_target.append(REF_CLASS)

    print("finished {}".format(i))

print(" LGG Train")
REF_CLASS = 1
for i in survival_id_tr_LGG:
    for j in data_types:
        img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = transform.resize(img, output_shape=(28, 28, CHANNELS))
        img = img.astype(np.float32)
        X_train_input.append(img)
        y_train_target.append(REF_CLASS)
    print("finished {}".format(i))

X_train_input = np.asarray(X_train_input, dtype=np.float32)
y_train_target = np.asarray(y_train_target, dtype=np.uint8)

print(X_train_input.shape)
print(y_train_target.shape)
