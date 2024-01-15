import argparse
import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable

from DataOp import TrainGenerator, get_list
from Network import MixAttNet
from Utils import check_dir, AvgMeter, dice_score, EarlyStopping


def argparser():
    parser = argparse.ArgumentParser(
        description="Training for Multi-class Segmentation with Deep Attentive CNN"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning Rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=int,
        default=1e-4,
        help="Weight Decay (default: 1e-4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch Size (default: 1)",
    )
    parser.add_argument(
        "--num_class",
        type=int,
        default=5,
        help="Number of classes (default: 5)",
    )
    parser.add_argument(
        "--num_iteration",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--val_fre",
        type=int,
        default=2,
        help="Validation Frequency (default: 2)",
    )
    parser.add_argument(
        "--pre_fre",
        type=int,
        default=10,
        help="Training loss print frequency (default: 10)",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Patience for early stopping (default: 50)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Patch Size (default: 128)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='Data/',
        help="Data Path (default: 'Data/')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='output/',
        help="Output Path (default: 'output/')",
    )
    return parser.parse_args()

def adjust_lr(lr, optimizer, iteration, num_iteration):
    """
    we decay the learning rate by a factor of 0.1 in 1/2 and 3/4 of whole training process
    """
    if iteration == num_iteration // 2:
        new_lr = lr*0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    elif iteration == num_iteration // 4 * 3:
        new_lr = lr*0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        pass


def loss_func(predict, label):
    """
    here we define the loss function, which you can upload additional loss in here
    """
    bce_loss = F.cross_entropy(predict, label)
    return bce_loss


def train_batch(net, optimizer, loader, num_class, device, output_path, iteration, pre_fre, train_loss_list):
    net.train()

    image, label = loader.get_item()

    image = Variable(torch.from_numpy(image).to(device))
    label = Variable(torch.from_numpy(label).to(device))

    predict = net(image)

    optimizer.zero_grad()

    label_one_hot = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=num_class)
    label_one_hot = np.squeeze(torch.swapaxes(label_one_hot, 1, 5), axis=5).float()

    loss1 = loss_func(predict[0], label_one_hot)
    loss2 = loss_func(predict[1], label)
    loss3 = loss_func(predict[2], label)
    loss4 = loss_func(predict[3], label)
    loss5 = loss_func(predict[4], label)
    loss6 = loss_func(predict[5], label)
    loss7 = loss_func(predict[6], label)
    loss8 = loss_func(predict[7], label)
    loss9 = loss_func(predict[8], label)
    loss = loss1 + \
           0.8*loss2 + 0.7*loss3 + 0.6*loss4 + 0.5*loss5 + \
           0.8*loss6 + 0.7*loss7 + 0.6*loss8 + 0.5*loss9
    loss.backward()
    optimizer.step()

    if iteration % pre_fre == 0:
        # Save training loss
        train_loss_list.append(loss.item())

        # Save training images, prediction and ground truth mask
        save_path = os.path.join(output_path, 'train_images')
        check_dir(save_path)

        image = np.squeeze(image)
        label = np.squeeze(label)
        predict = np.squeeze(torch.softmax(predict[0],  dim=1))
        predict = torch.argmax(predict,  dim=0)

        image_nii = nib.Nifti1Image(image.cpu().data.numpy(), affine=None)
        label_nii = nib.Nifti1Image(label.cpu().data.numpy(), affine=None)
        predict_nii = nib.Nifti1Image(predict.cpu().data.numpy(), affine=None)

        check_dir(os.path.join(save_path, '{}'.format(iteration)))
        nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(iteration)))
        nib.save(label_nii, os.path.join(save_path, '{}/label.nii.gz'.format(iteration)))
        nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(iteration)))

    return loss.item()


def val(net, loader, num_class, device, output_path, iteration, val_loss_list):
    net.eval()
    metric_meter = AvgMeter()

    image, label = loader.get_item()

    image = Variable(torch.from_numpy(image).to(device))
    label = Variable(torch.from_numpy(label).to(device))

    predict = net(image)

    label_one_hot = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=num_class)
    label_one_hot = np.squeeze(torch.swapaxes(label_one_hot, 1, 5), axis=5).float()
    metric_meter.update(dice_score(label_one_hot, predict, num_class))

    # Save val loss
    val_loss_list.append(metric_meter.avg)

    # Save validation images and prediction
    save_path = os.path.join(output_path, 'val_images')
    check_dir(save_path)

    image = np.squeeze(image)
    label = np.squeeze(label)
    predict = np.squeeze(torch.softmax(predict, dim=1))
    predict = torch.argmax(predict,  dim=0)

    image_nii = nib.Nifti1Image(image.cpu().data.numpy(), affine=None)
    label_nii = nib.Nifti1Image(label.cpu().data.numpy(), affine=None)
    predict_nii = nib.Nifti1Image(predict.cpu().data.numpy(), affine=None)

    check_dir(os.path.join(save_path, '{}'.format(iteration)))
    nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(iteration)))
    nib.save(label_nii, os.path.join(save_path, '{}/label.nii.gz'.format(iteration)))
    nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(iteration)))

    return metric_meter.avg


def main():
    args = argparser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    check_dir(args.output_path)
    ckpt_path = os.path.join(args.output_path, "ckpt")
    check_dir(ckpt_path)

    train_list, val_list, test_list = get_list(dir_path=args.data_path)

    train_generator = TrainGenerator(train_list,
                                     batch_size=args.batch_size,
                                     patch_size=args.patch_size)
    val_generator = TrainGenerator(val_list,
                                     batch_size=args.batch_size,
                                     patch_size=args.patch_size)
    net = MixAttNet(num_class=args.num_class).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True, path=os.path.join(ckpt_path, "early_stopping.pth.gz"))

    open(os.path.join(args.output_path, "train_record.txt"), 'w+')

    loss_meter = AvgMeter()
    start_time = time.time()
    best_metric = 0.

    train_loss_list = []
    pre_fre_epochs = []
    val_loss_list = []
    val_fre_epochs = []

    for iteration in range(1, args.num_iteration+1):
        adjust_lr(args.lr, optimizer, iteration, args.num_iteration)
        train_loss = train_batch(net=net, optimizer=optimizer, loader=train_generator, patch_size=args.patch_size, 
                                 batch_size=args.batch_size, device=device, num_class=args.num_class, output_path=args.output_path, 
                                 iteration=iteration, pre_fre=args.pre_fre, train_loss_list=train_loss_list)
        loss_meter.update(train_loss)

        if iteration % args.pre_fre == 0:
            pre_fre_epochs.append(iteration)
            iteration_time = time.time() - start_time
            info = [iteration, loss_meter.avg, iteration_time]
            print("Iter[{}] | Loss: {:.3f} | Time: {:.2f}".format(*info))
            start_time = time.time()
            loss_meter.reset()

        if iteration % args.val_fre == 0:
            val_fre_epochs.append(iteration)
            val_dice = val(net, loader=val_generator, num_class=args.num_class, device=device, output_path=args.output_path, 
                           iteration=iteration, val_loss_list=val_loss_list)
            if val_dice > best_metric:
                torch.save(net.state_dict(), os.path.join(ckpt_path, "best_val.pth.gz"))
                best_metric = val_dice
            open(os.path.join(args.output_path, "train_record.txt"), 'a+').write("{:.3f} | {:.3f}\n".format(train_loss, val_dice))
            print("Val in Iter[{}] Dice: {:.3f}".format(iteration, val_dice))
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_dice, net)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if iteration % 100 == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_path, "train_{}.pth.gz".format(iteration)))

    # Plot train and val loss and save figure
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.title("Train Loss")
    plt.xlabel('epochs')
    plt.ylabel("Cross Entropy Loss")
    plt.plot(pre_fre_epochs, train_loss_list)
    plt.subplot(2, 1, 2)
    plt.title("Validation Metric Meter")
    plt.xlabel('epochs')
    plt.ylabel("Dice Score")
    plt.plot(val_fre_epochs, val_loss_list)
    plt.savefig(os.path.join(args.output_path, "Train_val_loss.png"))
    plt.show()

if __name__ == '__main__':
    main()
