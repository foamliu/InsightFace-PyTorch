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
                bbox = tokens[4]

                filename = 'megaface/FaceScrub/{0}/{0}_{1}.jpg'.format(name, face_id)
                if os.path.isfile(filename):
                    samples.append({'name': name, 'face_id': face_id, 'bbox': bbox})

    print(len(samples))
