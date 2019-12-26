import os
from collections import Counter
if __name__ == "__main__":
    cnt = Counter()
    for dirName, subdirList, fileList in os.walk('megaface/FaceScrub'):
        for fname in fileList:
            name, ext = os.path.splitext(fname)
            cnt[ext] += 1
    print(cnt)


    annotation_files = ['facescrub_actors.txt', 'facescrub_actresses.txt']

    samples = []

    for anno in annotation_files:
        anno_file = os.path.join('megaface', anno)

        with open(anno_file, 'r') as fp:
            lines = fp.readlines()

            for line in lines:
                tokens = line.split('\t')
                name = tokens[0]
                face_id = tokens[2]
                bbox = tokens[4]
                for ext in ['jpg', 'jpeg', 'png']:
                    filename = 'megaface/FaceScrub/{0}/{0}_{1}.{2}'.format(name, face_id, ext)
                    if os.path.isfile(filename):
                        samples.append({'name': name, 'face_id': face_id, 'bbox': bbox, 'ext': ext})
                        break

    print(len(samples))
