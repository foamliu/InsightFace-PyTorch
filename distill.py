import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import device, print_freq
from data_gen import ArcFaceDataset
from megaface_eval import megaface_test
from mobilenet_v2 import MobileNetv2
from models import resnet101
from utils import parse_args, AverageMeter, get_logger

lr = 1e-5
batch_size = 256
num_workers = 8
end_epoch = 1000


def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * (
            T * T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, acc, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


def train_net(teacher_model):
    torch.manual_seed(7)
    np.random.seed(7)
    start_epoch = 0
    best_acc = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    model = MobileNetv2()
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    # Epochs
    for epoch in range(start_epoch, end_epoch):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           teacher_model=teacher_model,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)

        # One epoch's validation
        megaface_acc = megaface_test(model)
        writer.add_scalar('model/megaface_accuracy', megaface_acc, epoch)

        # Check if there was an improvement
        is_best = megaface_acc > best_acc
        best_acc = max(megaface_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_acc, is_best)


def train(train_loader, teacher_model, model, criterion, optimizer, epoch, logger):
    teacher_model.eval()
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, target) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        target = target.to(device)

        # Forward prop.
        output = model(img)  # embedding => [N, 512]
        with torch.no_grad():
            teacher_output = teacher_model(img)

        # Calculate loss
        # loss = criterion(output, teacher_output)  # class_id_out => [N, 10575]
        loss = criterion(output, target, teacher_output, T=20.0, alpha=0.7)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.6f} ({loss.avg:.6f})'.format(epoch, i, len(train_loader),
                                                                      loss=losses))

    return losses.avg


def main():
    global args
    args = parse_args()

    checkpoint = 'insight-face-v3.pt'
    print('loading model: {}...'.format(checkpoint))
    teacher_model = resnet101(args)
    teacher_model.load_state_dict(torch.load(checkpoint))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    train_net(teacher_model)


if __name__ == '__main__':
    main()
