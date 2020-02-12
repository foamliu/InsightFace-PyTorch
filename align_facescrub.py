import argparse
import os
import re
from multiprocessing import Pool

import cv2 as cv
import numpy as np
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

        img = cv.resize(img, (int(round(w * ratio)), int(round(h * ratio))), interpolation=cv.INTER_CUBIC)
    return img, ratio


def get_files():
    annotation_files = ['facescrub_actors.txt', 'facescrub_actresses.txt']

    samples = []

    for anno in annotation_files:
        anno_file = os.path.join('megaface', anno)

        with open(anno_file, 'r') as fp:
            lines = fp.readlines()

            for line in lines[1:]:
                tokens = line.split('\t')
                name = tokens[0]
                face_id = tokens[2]
                url = tokens[3]
                # print(url)
                ext = url.split('.')[-1]
                # print(ext)

                bbox = tokens[4]
                filename = '{0}/{0}_{1}.{2}'.format(name, face_id, ext)
                full_path = 'megaface/FaceScrub/{}'.format(filename)
                if os.path.isfile(full_path):
                    samples.append({'filename': filename, 'bbox': bbox})

    # print(len(samples))
    return samples


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def select_face(bboxes, boxB):
    max_iou = 0
    max_idx = 0

    for idx, boxA in enumerate(bboxes):
        iou = bb_intersection_over_union(boxA, boxB)
        # print(iou)

        if iou > max_iou:
            max_iou = max(iou, max_iou)
            max_idx = idx

    return max_idx


def detect_face(data):
    from retinaface.detector import detector
    from utils import align_face

    src_path = data['src_path']
    dst_path = data['dst_path']
    boxB = np.array(data['boxB'])

    img = cv.imread(src_path)
    if img is not None:
        img, ratio = resize(img)
        boxB = boxB * ratio

        try:
            bboxes, landmarks = detector.detect_faces(img)

            if len(bboxes) > 0:
                i = select_face(bboxes, boxB)
                bbox, landms = bboxes[i], landmarks[i]
                img = align_face(img, [landms])
                dirname = os.path.dirname(dst_path)
                os.makedirs(dirname, exist_ok=True)
                cv.imwrite(dst_path, img)
        except ValueError as err:
            print(err)
        except cv.error as err:
            print(err)

    return True


def align_facescrub(src, dst):
    image_paths = []
    for sample in get_files():
        fname = sample['filename']
        boxB = eval(sample['bbox'])
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname).replace(' ', '_')
        pattern = re.compile(re.escape('.png'), re.IGNORECASE)
        dst_path = pattern.sub('.jpg', dst_path)
        image_paths.append({'src_path': src_path, 'dst_path': dst_path, 'boxB': boxB})

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
    parser.add_argument('--src', type=str, default='megaface/FaceScrub', help='src path')
    parser.add_argument('--dst', type=str, default='megaface/FaceScrub_aligned', help='dst path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    src = args.src
    dst = args.dst

    align_facescrub(src, dst)

    # python3 align_facescrub.py --src megaface/FaceScrub --dst megaface/FaceScrub_aligned
