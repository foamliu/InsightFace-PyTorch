import subprocess

if __name__ == '__main__':
    cmd = 'find megaface/facescrub_images -name "*.bin" -type f -delete'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    cmd = 'find megaface/MegaFace_aligned/FlickrFinal2 -name "*.bin" -type f -delete'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    cmd = 'python3 megaface.py --action gen_features'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    cmd = 'python3 megaface.py --action gen_features'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    cmd = 'python run_experiment.py -p /dev/code/mnt/InsightFace-v3/megaface/devkit/templatelists/facescrub_uncropped_features_list.json /dev/code/mnt/InsightFace-v3/megaface/MegaFace_aligned/FlickrFinal2 /dev/code/mnt/InsightFace-v3/megaface/facescrub_images _0.bin results -s 1000000'
    print(cmd)
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    print(output)

    lines = output.split('\n')
    line = [l for l in lines if l.startswith('Rank 1: ')][0]
    accuracy = float(line[8:])
    print(accuracy)
