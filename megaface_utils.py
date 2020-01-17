import argparse
import json
import os
import struct

import cv2 as cv
import numpy as np
import torch
import tqdm
from PIL import Image, ImageOps
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import align_face, get_central_face_attributes


def walkdir(folder, ext):
    # Walk through each files in a directory
    for dirpath, dirs, files in os.walk(folder):
        for filename in [f for f in files if f.lower().endswith(ext)]:
            yield os.path.abspath(os.path.join(dirpath, filename))


def crop_one_image(filepath, oldkey, newkey):
    new_fn = filepath.replace(oldkey, newkey)
    tardir = os.path.dirname(new_fn)
    if not os.path.isdir(tardir):
        os.makedirs(tardir)

    if not os.path.exists(new_fn):
        is_valid, bounding_boxes, landmarks = get_central_face_attributes(filepath)
        if is_valid:
            img = align_face(filepath, landmarks)
            cv.imwrite(new_fn, img)


def crop(path, oldkey, newkey):
    print('Counting images under {}...'.format(path))
    # Preprocess the total files count
    filecounter = 0
    for filepath in walkdir(path, '.jpg'):
        filecounter += 1

    for filepath in tqdm(walkdir(path, '.jpg'), total=filecounter, unit="files"):
        crop_one_image(filepath, oldkey, newkey)

    print('{} images were cropped successfully.'.format(filecounter))


def get_image(transformer, filepath, flip=False):
    img = Image.open(filepath)
    if flip:
        img = ImageOps.flip(img)
    img = transformer(img)
    return img.to(device)


def gen_feature(path, model):
    model.eval()

    print('gen features {}...'.format(path))
    # Preprocess the total files count
    files = []
    for filepath in walkdir(path, ('.jpg', '.png')):
        files.append(filepath)
    file_count = len(files)

    transformer = data_transforms['val']

    batch_size = 128

    with torch.no_grad():
        for start_idx in tqdm(range(0, file_count, batch_size)):
            end_idx = min(file_count, start_idx + batch_size)
            length = end_idx - start_idx

            imgs_0 = torch.zeros([length, 3, 112, 112], dtype=torch.float, device=device)
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                imgs_0[idx] = get_image(transformer, filepath, flip=False)

            features_0 = model(imgs_0.to(device))
            features_0 = features_0.cpu().numpy()

            imgs_1 = torch.zeros([length, 3, 112, 112], dtype=torch.float, device=device)
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                imgs_1[idx] = get_image(transformer, filepath, flip=True)

            features_1 = model(imgs_1.to(device))
            features_1 = features_1.cpu().numpy()

            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                filepath = filepath.replace(' ', '_')
                tarfile = filepath + '_0.bin'
                feature = features_0[idx] + features_1[idx]
                write_feature(tarfile, feature / np.linalg.norm(feature))


def read_feature(filename):
    f = open(filename, 'rb')
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4 * 4))
    mat = np.fromstring(f.read(rows * 4), dtype=np.dtype('float32'))
    return mat.reshape(rows, 1)


def write_feature(filename, m):
    header = struct.pack('iiii', m.shape[0], 1, 4, 5)
    f = open(filename, 'wb')
    f.write(header)
    f.write(m.data)


def remove_noise():
    megaface_count = 0
    for line in open('megaface/megaface_noises.txt', 'r'):
        filename = 'megaface/MegaFace_aligned/FlickrFinal2/' + line.strip() + '_0.bin'
        if os.path.exists(filename):
            # print(filename)
            os.remove(filename)
            megaface_count += 1

    print('remove noise - megaface: ' + str(megaface_count))

    facescrub_count = 0
    noise = set()
    for line in open('megaface/facescrub_noises.txt', 'r'):
        noise.add((line.strip().replace('.png', '.jpg') + '_0.bin'))

    for root, dirs, files in os.walk('megaface/FaceScrub_aligned'):
        for f in files:
            # print(f)
            if f in noise:
                filename = os.path.join(root, f)
                if os.path.exists(filename):
                    # print(filename)
                    os.remove(filename)
                    facescrub_count += 1

    print('remove noise - facescrub: ' + str(facescrub_count))


def test():
    root1 = '/root/lin/data/FaceScrub_aligned/Benicio Del Toro'
    root2 = '/root/lin/data/FaceScrub_aligned/Ben Kingsley'
    for f1 in os.listdir(root1):
        for f2 in os.listdir(root2):
            if f1.lower().endswith('.bin') and f2.lower().endswith('.bin'):
                filename1 = os.path.join(root1, f1)
                filename2 = os.path.join(root2, f2)
                fea1 = read_feature(filename1)
                fea2 = read_feature(filename2)
                print(((fea1 - fea2) ** 2).sum() ** 0.5)


def match_result():
    with open('matches_facescrub_megaface_0_1000000_1.json', 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
        for i in range(len(load_dict)):
            print(load_dict[i]['probes'])


def pngtojpg(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1] == '.png':
                img = cv.imread(os.path.join(root, f))
                newfilename = f.replace(".png", ".jpg")
                cv.imwrite(os.path.join(root, newfilename), img)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--action', default='crop_megaface', help='action')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.action == 'crop_megaface':
        crop('megaface/MegaFace/FlickrFinal2', 'MegaFace', 'MegaFace_aligned')
    elif args.action == 'crop_facescrub':
        crop('megaface/facescrub_images', 'facescrub', 'facescrub_aligned')
    elif args.action == 'gen_features':
        gen_feature('megaface/facescrub_images')
        gen_feature('megaface/MegaFace_aligned/FlickrFinal2')
        remove_noise()
    elif args.action == 'pngtojpg':
        pngtojpg('megaface/facescrub_images')
    elif args.action == 'remove_noise':
        remove_noise()
