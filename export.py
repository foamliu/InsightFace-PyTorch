import torch


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    print(type(model))
    print('use_se: ' + str(model.use_se))
    print('fc: ' + str(model.fc))
    print('layer1: ' + str(model.layer1))
    print('layer2: ' + str(model.layer2))
    print('layer3: ' + str(model.layer3))
    print('layer4: ' + str(model.layer4))

    # model.eval()

    torch.save(model.state_dict(), 'insight-face-v3.pt')
