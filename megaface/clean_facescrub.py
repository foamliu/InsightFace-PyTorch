from tqdm import tqdm
import os
import shutil

if __name__ == "__main__":
    folder = 'FaceScrub'

    folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for d in folders:
        src = os.path.join(folder, d)
        dst = os.path.join(folder, d.replace('\'', ''))
        print('{} -> {}'.format(src, dst))
        shutil.move(src, dst)

    # image_paths = []
    # for dirName, subdirList, fileList in tqdm(os.walk(folder)):
    #     for fname in fileList:
    #         if fname.lower().endswith('.jpg'):
    #             src_path = os.path.join(dirName, fname)
    #             dst_path = os.path.join(dirName.replace(src, dst), fname)
    #             image_paths.append({'src_path': src_path, 'dst_path': dst_path})