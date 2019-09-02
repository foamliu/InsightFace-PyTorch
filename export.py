import torch


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    # model.eval()

    torch.save(model.state_dict(), 'insight-face-v3.pt')
