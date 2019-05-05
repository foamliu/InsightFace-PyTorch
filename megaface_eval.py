import os
import subprocess

if __name__ == '__main__':
    cmd = 'find megaface/facescrub_images -name "*.bin" -type f -delete'
    result = subprocess.run([cmd], stdout=subprocess.PIPE)
    print(result.stdout)

    cmd = 'find megaface/MegaFace_aligned/FlickrFinal2 -name "*.bin" -type f -delete'
    result = subprocess.run([cmd], stdout=subprocess.PIPE)
    print(result.stdout)

    cmd = 'python3 megaface.py --action gen_features'
    result = subprocess.run([cmd], stdout=subprocess.PIPE)
    print(result.stdout)

    cmd = 'python3 megaface.py --action gen_features'
    result = subprocess.run([cmd], stdout=subprocess.PIPE)
    print(result.stdout)

    cmd = 'python run_experiment.py -p /dev/code/mnt/InsightFace-v3/megaface/devkit/templatelists/facescrub_uncropped_features_list.json /dev/code/mnt/InsightFace-v3/megaface/MegaFace_aligned/FlickrFinal2 /dev/code/mnt/InsightFace-v3/megaface/facescrub_images _0.bin results -s 1000000'
    result = subprocess.run([cmd], stdout=subprocess.PIPE)
    print(result.stdout)