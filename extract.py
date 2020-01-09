import os

import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('data')
    zip_ref.close()


if __name__ == "__main__":
    if not os.path.isdir('data/faces_emore'):
        extract('data/faces_emore.zip')
