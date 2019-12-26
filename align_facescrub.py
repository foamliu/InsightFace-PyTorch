import os

if __name__ == "__main__":
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
                url = tokens[3]
                ext = url.split('.')[-1]
                print(ext)
                break
                bbox = tokens[4]
                filename = '{0}_{1}.{2}'.format(name, face_id, ext)
                full_path = 'megaface/FaceScrub/{0}/{1}'.format(name, filename)
                if os.path.isfile(filename):
                    samples.append({'name': name, 'face_id': face_id, 'bbox': bbox, 'ext': ext})
                    break
        break

    print(len(samples))
