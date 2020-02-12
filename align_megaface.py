import argparse
import os
from multiprocessing import Pool

import cv2 as cv
import tqdm
from tqdm import tqdm


def resize(img):
    max_size = 800
    ratio = 1
    h, w = img.shape[:2]

    if h > max_size or w > max_size:
        if h > w:
            ratio = max_size / h
        else:
            ratio = max_size / w

        img = cv.resize(img, (int(round(w * ratio)), int(round(h * ratio))))
    return img, ratio


def detect_face(data):
    from retinaface.detector import detector
    from utils import align_face

    src_path = data['src_path']
    dst_path = data['dst_path']
    # print(src_path)

    img_raw = cv.imread(src_path)
    if img_raw is not None:
        img, _ = resize(img_raw)

        try:
            bboxes, landmarks = detector.detect_faces(img)

            if len(bboxes) > 0:
                bbox, landms = bboxes[0], landmarks[0]
                img = align_face(img, [landms])
                dirname = os.path.dirname(dst_path)
                os.makedirs(dirname, exist_ok=True)
                cv.imwrite(dst_path, img)
                return True

        except ValueError as err:
            print(err)

    return False


def align_megaface(src, dst):
    image_paths = []
    for dirName, subdirList, fileList in tqdm(os.walk(src)):
        for fname in fileList:
            if fname.lower().endswith(('.jpg', '.png')):
                src_path = os.path.join(dirName, fname)
                dst_path = os.path.join(dirName.replace(src, dst), fname).replace(' ', '_')
                image_paths.append({'src_path': src_path, 'dst_path': dst_path})

    # print(image_paths[:20])
    num_images = len(image_paths)
    print('num_images: ' + str(num_images))

    with Pool(4) as p:
        r = list(tqdm(p.imap(detect_face, image_paths), total=num_images))

    # for image_path in tqdm(image_paths):
    #     detect_face(image_path)

    print('Completed!')


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--src', type=str, default='megaface/MegaFace', help='src path')
    parser.add_argument('--dst', type=str, default='megaface/MegaFace_aligned', help='dst path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    src = args.src
    dst = args.dst

    align_megaface(src, dst)

    # python3 align_megaface.py --src megaface/MegaFace --dst megaface/MegaFace_aligned
