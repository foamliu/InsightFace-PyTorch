import subprocess

from megaface_utils import gen_feature, remove_noise


def megaface_test(model):
    cmd = 'find megaface/facescrub_images -name "*.bin" -type f -delete'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    cmd = 'find megaface/MegaFace_aligned/FlickrFinal2 -name "*.bin" -type f -delete'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    gen_feature('megaface/facescrub_images', model)
    gen_feature('megaface/MegaFace_aligned/FlickrFinal2', model)
    remove_noise()

    cmd = 'python megaface/devkit/experiments/run_experiment.py -p megaface/devkit/templatelists/facescrub_uncropped_features_list.json megaface/MegaFace_aligned/FlickrFinal2 megaface/facescrub_images _0.bin results -s 1000000'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    lines = output.split('\n')
    line = [l for l in lines if l.startswith('Rank 1: ')][0]
    accuracy = float(line[8:])
    return accuracy


if __name__ == '__main__':
    megaface_test(None)
