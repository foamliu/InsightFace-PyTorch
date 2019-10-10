import os
from tqdm import tqdm

def detect_face():
    pass


def megaface_align(src, dst):
    image_paths = []
    for dirName, subdirList, fileList in tqdm(os.walk(src)):
        for fname in fileList:
            if fname.lower().endswith('.jpg'):
                image_paths.append({'dirName':dirName, 'fname':fname})

    print(image_paths[:20])
    print(len(image_paths))


    # with Pool(2) as p:
    #     r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))


if __name__ == '__main__':
    src = 'megaface/MegaFace'
    dst = 'megaface/MegaFace_aligned'
    megaface_align(src, dst)
