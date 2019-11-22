import argparse
import os

import cv2 as cv
import torch
import tqdm
from PIL import Image
from torch.multiprocessing import Pool
from tqdm import tqdm


def get_central_face_attributes(full_path):
    from retinaface.detector import detect_faces
    from utils import select_central_face
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            i = select_central_face(img.size, bounding_boxes)
            return True, [bounding_boxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except ValueError:
        pass
    except IOError:
        pass
    return False, None, None


def detect_face(data):
    from utils import align_face

    src_path = data['src_path']
    dst_path = data['dst_path']
    with torch.no_grad():
        has_face, bboxes, landmarks = get_central_face_attributes(src_path)
        if has_face:
            img = align_face(src_path, landmarks)
            cv.imwrite(dst_path, img)
            print(dst_path)

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

    with Pool(2) as p:
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

    megaface_align(src, dst)

    # megaface_align('megaface/MegaFace', 'megaface/MegaFace_aligned')
    # megaface_align('megaface/FaceScrub', 'megaface/FaceScrub_aligned')

    # CUDA_VISIBLE_DEVICES=0 python3 megaface_align.py --src megaface/MegaFace --dst megaface/MegaFace_aligned
    # CUDA_VISIBLE_DEVICES=1 python3 megaface_align.py --src megaface/FaceScrub --dst megaface/FaceScrub_aligned
