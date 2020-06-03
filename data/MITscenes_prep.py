"""
Prepare the MIT indoor scene recognition dataset: http://web.mit.edu/torralba/www/indoor.html
The 5 supercategories are used as the 5 tasks.
"""
import subprocess
import os
import csv
from distutils.dir_util import copy_tree
import shutil

import shared_utils.utils as utils


def download_dset(path):
    utils.create_dir(path)

    if not os.path.exists(os.path.join(path, 'indoorCVPR_09.tar')):
        subprocess.call(
            "wget -P {} http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar".format(path),
            shell=True)
        print("Succesfully downloaded MIT dataset.")

    if not os.path.exists(os.path.join(path, 'Images')):
        subprocess.call(
            "tar -C {} -xvf {}".format(path, os.path.join(path, 'indoorCVPR_09.tar')),
            shell=True)
        print("Succesfully extracted MIT dataset.")

    if not os.path.exists(os.path.join(path, 'TrainImages.txt')):
        subprocess.call(
            "wget -P {} http://web.mit.edu/torralba/www/TrainImages.txt".format(path),
            shell=True)
        print("Succesfully downloaded MIT train img list.")
    if not os.path.exists(os.path.join(path, 'TestImages.txt')):
        subprocess.call(
            "wget -P {} http://web.mit.edu/torralba/www/TestImages.txt".format(path),
            shell=True)
        print("Succesfully downloaded MIT test img list.")


def split_train_eval(root_path):
    """
    Split in train (TrainImages) and eval subsets (TestImages).
    Not all imges are included, so the extra data resides in 'Images'.
    """
    print("Splitting into train/test images.")
    tr_list = os.path.join(root_path, "TrainImages.txt")
    move_from_list(tr_list, root_path, 'TrainImages')

    test_list = os.path.join(root_path, "TestImages.txt")
    move_from_list(test_list, root_path, 'TestImages')


def move_from_list(img_list, root_path, subdir):
    img_path = os.path.join(root_path, "Images")

    with open(img_list) as f:
        file_paths = f.readlines()
    file_paths = [x.strip() for x in file_paths]

    for file_path in file_paths:
        src_file = os.path.join(img_path, file_path)

        supercat, filename = os.path.split(file_path)
        target_path = os.path.join(root_path, subdir, supercat)
        utils.create_dir(target_path)

        target_file = os.path.join(target_path, filename)
        if not os.path.exists(target_file):
            shutil.move(src_file, target_file)


def split_tasks(root_path, imgsubset):
    """ Split into <root>/Tasks/<imgsubset>/<supercategory>/<category>/<imges>"""
    supercat_dict = parse_supercat_mapping()

    img_path = os.path.join(root_path, imgsubset)
    for supercat, cat_list in supercat_dict.items():
        for cat in cat_list:
            task_dir = os.path.join(root_path, 'Tasks', imgsubset, supercat)
            utils.create_dir(task_dir)
            target_dir = os.path.join(task_dir, cat)
            if not os.path.exists(target_dir):
                copy_tree(os.path.join(img_path, cat), target_dir)


def parse_supercat_mapping():
    supercat_dict = {}
    csvpath = os.path.join(os.path.dirname(__file__), 'MITscenes_supercategory_mapping.csv')
    with open(csvpath, newline='') as csvfile:
        mapping = csv.reader(csvfile, quotechar='|')

        for row in mapping:
            if len(row) == 2:
                if row[0] not in supercat_dict:
                    supercat_dict[row[0]] = [row[1]]
                else:
                    supercat_dict[row[0]].append(row[1])

    total_cat = 0
    for supercat, cat in supercat_dict.items():
        print("{}: {} categories".format(supercat, len(cat)))
        total_cat += len(cat)

    print(supercat_dict)
    assert len(supercat_dict.keys()) == 5
    assert total_cat == 67

    return supercat_dict


def main():
    config = utils.get_parsed_config()
    parent_path = utils.read_from_config(config, 'ds_root_path')
    root_path = os.path.join(parent_path, "MIT_indoor_scenes")

    download_dset(root_path)

    split_train_eval(root_path)

    print("Splitting into tasks.")
    for x in ['Images', 'TrainImages', 'TestImages']:
        print("SPLITTING INTO TASKS for dir {}".format(x))
        split_tasks(root_path, x)

    print("Done all MIT indoor scenes data processing.")


if __name__ == '__main__':
    main()
