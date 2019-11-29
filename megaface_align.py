import argparse
import os

import cv2 as cv
import tqdm
from torch.multiprocessing import Pool
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


def detect_face(data):
    from retinaface.detector import detect_faces
    from utils import select_significant_face, align_face

    src_path = data['src_path']
    dst_path = data['dst_path']

    img_raw = cv.imread(src_path)
    if img_raw is not None:
        img = resize(img_raw)
        bboxes, landmarks = detect_faces(img, top_k=5, keep_top_k=5)
        if len(bboxes) > 0:
            i = select_significant_face(bboxes)
            bbox, landms = bboxes[i], landmarks[i]
            img = align_face(img, [landms])
            dirname = os.path.dirname(dst_path)
            os.makedirs(dirname, exist_ok=True)
            cv.imwrite(dst_path, img)

    return True


def megaface_align(src, dst, size):
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

    with Pool(size) as p:
        r = list(tqdm(p.imap(detect_face, image_paths), total=num_images))

    # for image_path in tqdm(image_paths):
    #     detect_face(image_path)

    print('Completed!')


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--src', type=str, default='megaface/MegaFace', help='src path')
    parser.add_argument('--dst', type=str, default='megaface/MegaFace_aligned', help='dst path')
    parser.add_argument('--size', type=int, default=4, help='processes')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    src = args.src
    dst = args.dst
    size = args.size

    megaface_align(src, dst, size)

    # megaface_align('megaface/MegaFace', 'megaface/MegaFace_aligned')
    # megaface_align('megaface/FaceScrub', 'megaface/FaceScrub_aligned')

    # CUDA_VISIBLE_DEVICES=0 python3 megaface_align.py --src megaface/MegaFace --dst megaface/MegaFace_aligned
    # CUDA_VISIBLE_DEVICES=1 python3 megaface_align.py --src megaface/FaceScrub --dst megaface/FaceScrub_aligned --pool 1
