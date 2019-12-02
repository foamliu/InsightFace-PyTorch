import os
import pickle

import cv2 as cv
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm

from config import path_imgidx, path_imgrec, IMG_DIR, pickle_file
from utils import ensure_folder

if __name__ == "__main__":
    ensure_folder(IMG_DIR)
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    samples = []
    class_ids = set()

    # %% 1 ~ 5822653
    for i in tqdm(range(5822653)):
        try:
            header, s = recordio.unpack(imgrec.read_idx(i + 1))
            img = mx.image.imdecode(s).asnumpy()
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            label = int(header.label)
            class_ids.add(label)
            filename = '{}.jpg'.format(i)
            samples.append({'img': filename, 'label': label})
            filename = os.path.join(IMG_DIR, filename)
            cv.imwrite(filename, img)
        except KeyboardInterrupt:
            raise
        except:
            print(i)
            break

    with open(pickle_file, 'wb') as file:
        pickle.dump(samples, file)

    print('num_samples: ' + str(len(samples)))

    class_ids = list(class_ids)
    print(len(class_ids))
    print(max(class_ids))
