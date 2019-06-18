import json
import os


def match_result():
    with open('results/matches_facescrub_megaface_0_1000000_1.json', 'r') as file:
        data = json.load(file)

    print(len(data))
    num_total = 0
    num_incorrect = 0

    # # print(load_dict)
    for i, item in enumerate(data):
        probes = item['probes']
        idx = probes['idx']
        rank = probes['rank']
        incorrect = []
        for j in range(len(idx)):
            if rank[j] > 0:
                incorrect.append(str(idx[j]))
        if len(incorrect) > 0:
            print(i)
            print(' '.join(incorrect))
        num_total += len(idx)
        num_incorrect += len(incorrect)

    print(num_incorrect)
    print(num_total)
    print(num_incorrect / num_total)


def check_facescrub():
    num_bins = 0
    folder = 'facescrub_images'
    dir_list = [d for d in os.listdir(folder)]
    for d in dir_list:
        str_dir = os.path.join(folder, d)
        bin_list = [f for f in os.listdir(str_dir) if f.endswith('.bin')]
        num_bins += len(bin_list)
    print('num_bins: ' + str(num_bins))


if __name__ == '__main__':
    match_result()
    # check_facescrub()
