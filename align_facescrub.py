import argparse
import os
from multiprocessing import Pool

import cv2 as cv
import tqdm
from tqdm import tqdm


def resize(img):
    max_size = 600
    h, w = img.shape[:2]
    if h <= max_size and w <= max_size:
        return img
    if h > w:
        ratio = max_size / h
    else:
        ratio = max_size / w

    img = cv.resize(img, (int(round(w * ratio)), int(round(h * ratio))))
    return img


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


def detect_face(data):
    from utils import select_significant_face, align_face
    from retinaface.detector import detector

    src_path = data['src_path']
    dst_path = data['dst_path']

    img_raw = cv.imread(src_path)
    if img_raw is not None:
        img = resize(img_raw)
        try:
            bboxes, landmarks = detector.detect_faces(img, confidence_threshold=0.9)

            if len(bboxes) > 0:
                i = select_significant_face(bboxes)
                bbox, landms = bboxes[i], landmarks[i]
                img = align_face(img, [landms])
                dirname = os.path.dirname(dst_path)
                os.makedirs(dirname, exist_ok=True)
                cv.imwrite(dst_path, img)

        except ValueError as err:
            print(err)

    return True


def align_megaface(src, dst, size):



    image_paths = []
    for dirName, subdirList, fileList in tqdm(os.walk(src)):
        for fname in fileList:
            if fname.lower().endswith(('.jpg', '.png')):
                src_path = os.path.join(dirName, fname)
                dst_path = os.path.join(dirName.replace(src, dst), fname).replace(' ', '_')
                image_paths.append({'src_path': src_path, 'dst_path': dst_path})

    # # print(image_paths[:20])
    # num_images = len(image_paths)
    # print('num_images: ' + str(num_images))
    #
    # with Pool(size) as p:
    #     r = list(tqdm(p.imap(detect_face, image_paths), total=num_images))
    #
    # # for image_path in tqdm(image_paths):
    # #     detect_face(detector, image_path)
    #
    # print('Completed!')


def parse_args():
    parser = argparse.ArgumentParser(description='align FaceScrub')
    # general
    parser.add_argument('--src', type=str, default='megaface/FaceScrub', help='src path')
    parser.add_argument('--dst', type=str, default='megaface/FaceScrub_aligned', help='dst path')
    parser.add_argument('--size', type=int, default=4, help='processes')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    src = args.src
    dst = args.dst
    size = args.size

    align_megaface(src, dst, size)

    # python3 align_facescrub.py --src megaface/FaceScrub --dst megaface/FaceScrub_aligned
