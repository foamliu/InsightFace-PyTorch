import os
from multiprocessing import Pool

import tqdm
from tqdm import tqdm

from utils import get_central_face_attributes, align_face


def detect_face(image_path):
    dirName = image_path['dirName']
    fname = image_path['fname']
    full_path = os.path.join(dirName, fname)
    has_face, bboxes, landmarks = get_central_face_attributes(full_path)
    if has_face:
        img = align_face(full_path, landmarks)
    return True


def megaface_align(src, dst):
    image_paths = []
    for dirName, subdirList, fileList in tqdm(os.walk(src)):
        for fname in fileList:
            if fname.lower().endswith('.jpg'):
                image_paths.append({'dirName': dirName, 'fname': fname})

    print(image_paths[:20])
    num_images = len(image_paths)
    print(num_images)

    with Pool(16) as p:
        r = list(tqdm(p.imap(detect_face, image_paths), total=num_images))

    print(r)


if __name__ == '__main__':
    src = 'megaface/MegaFace'
    dst = 'megaface/MegaFace_aligned'
    megaface_align(src, dst)
