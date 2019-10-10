import os


def detect_face():
    pass


def megaface_align(src, dst):
    for dirName, subdirList, fileList in os.walk(src):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            print('\t%s' % fname)

    # with Pool(2) as p:
    #     r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))


if __name__ == '__main__':
    src = 'megaface/MegaFace'
    dst = 'megaface/MegaFace_aligned'
    megaface_align(src, dst)
