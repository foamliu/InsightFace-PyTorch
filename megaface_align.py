import os

import cv2 as cv
import tqdm
from tqdm import tqdm

from utils import get_central_face_attributes, align_face


def detect_face(data):
    src_path = data['src_path']
    dst_path = data['dst_path']
    has_face, bboxes, landmarks = get_central_face_attributes(src_path)
    if has_face:
        img = align_face(src_path, landmarks)
        cv.imwrite(dst_path, img)

    return True


def megaface_align(src, dst):
    image_paths = []
    for dirName, subdirList, fileList in tqdm(os.walk(src)):
        for fname in fileList:
            if fname.lower().endswith('.jpg'):
                src_path = os.path.join(dirName, fname)
                dst_path = os.path.join(dirName.replace(src, dst), fname)
                image_paths.append({'src_path': src_path, 'dst_path': dst_path})

    # print(image_paths[:20])
    num_images = len(image_paths)
    print('num_images: ' + str(num_images))

    # with Pool(16) as p:
    #     r = list(tqdm(p.imap(detect_face, image_paths), total=num_images))

    for image_path in tqdm(image_paths):
        detect_face(image_path)

    print('Completed!')


if __name__ == '__main__':
    src = 'megaface/MegaFace'
    dst = 'megaface/MegaFace_aligned'
    megaface_align(src, dst)
