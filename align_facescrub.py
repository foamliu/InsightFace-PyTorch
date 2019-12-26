import os

if __name__ == "__main__":
    files = ['facescrub_actors.txt', 'facescrub_actresses.txt']

    samples = []

    for f in files:
        filename = os.path.join('megaface', f)

        with open(filename, 'r') as fp:
            lines = fp.readlines()

            for line in lines:
                tokens = line.split('\t')
                name = tokens[0]
                face_id = tokens[2]
                bbox = tokens[4]

                samples.append({'name': name, 'face_id': face_id, 'bbox': bbox})

    print(len(samples))
