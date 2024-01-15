import argparse
import os
import time

import nibabel as nib
import numpy as np

import torch
from torch.autograd import Variable

from Network import MixAttNet
from Utils import check_dir
from DataOp import get_list

def argparser():
    parser = argparse.ArgumentParser(
        description="Inference for Multi-Class Segmentaion with Deep Attentive CNN"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/output/ckpt/best_val.pth.gz",
        help="Path to weights file (default: '/output/ckpt/best_val.pth.gz')",
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
    parser.add_argument(
        "--num_class",
        type=int,
        default=5,
        help="Number of classes (default: 5)",
    )
    return parser.parse_args()

def main():
    """
    Inference on new images for instance segmentation on multi organs
    """
    args = argparser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_list, val_list, test_list = get_list(dir_path=args.data_path)
    net = MixAttNet(num_class=args.num_class).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device))

    save_path = os.path.join(args.output_path, 'test_images')
    check_dir(save_path)

    net.eval()

    for idx, data_dict in enumerate(test_list):
        image_path = data_dict['image_path']
        print("Start inference for image :", image_path)

        image = nib.load(image_path).get_fdata()
        image = Variable(torch.from_numpy(image[np.newaxis, np.newaxis, ...].astype(np.float32)).to(device))

        start_time = time.time()
        with torch.no_grad():
            predict = net(image) 

        image = np.squeeze(image)
        predict = np.squeeze(torch.softmax(predict, dim=1))
        predict = torch.argmax(predict, dim=0)

        image_nii = nib.Nifti1Image(image.cpu().data.numpy(), affine=None)
        predict_nii = nib.Nifti1Image(predict.cpu().data.numpy(), affine=None)

        check_dir(os.path.join(save_path, '{}'.format(idx)))
        nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(idx)))
        nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(idx)))
        print("Images Saved")

        print("[{}] Testing Finished, Cost {:.2f}s".format(idx, time.time()-start_time))


if __name__ == "__main__":
    main()