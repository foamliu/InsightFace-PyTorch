import time

import torch

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    print(type(model))
    print('use_se: ' + str(model.use_se))
    print('fc: ' + str(model.fc))
    # print('layer1: ' + str(model.layer1))
    # print('layer2: ' + str(model.layer2))
    # print('layer3: ' + str(model.layer3))
    # print('layer4: ' + str(model.layer4))

    # model.eval()
    filename = 'insight-face-v3.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))


    class HParams:
        def __init__(self):
            self.pretrained = False
            self.use_se = True


    config = HParams()

    print('loading {}...'.format(filename))
    start = time.time()
    from models import resnet101

    model = resnet101(config)
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
