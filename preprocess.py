""" preprocess """
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset, cifar10')
parser.add_argument('--dataset_path', type=str, default="../cifar-10/cifar-10-verify-bin",
                    help='Dataset path.')
parser.add_argument('--output_path', type=str, default="./preprocess_Result",
                    help='preprocess Result path.')
args_opt = parser.parse_args()

# import dataset
if args_opt.dataset == "cifar10":
    from src.dataset import create_dataset1 as create_dataset
    from src.config import config1 as config
elif args_opt.dataset == "cifar100":
    from src.dataset import create_dataset2 as create_dataset
    from src.config import config2 as config
else:
    raise ValueError("dataset is not support.")


def get_cifar_bin():
    '''generate cifar bin files.'''
    ds = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, batch_size=config.batch_size)
    img_path = os.path.join(args_opt.output_path, "00_img_data")
    label_path = os.path.join(args_opt.output_path, "label.npy")
    os.makedirs(img_path)
    label_list = []

    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True)):
        img_data = data["image"]
        img_label = data["label"]

        file_name = args_opt.dataset + "_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)
        label_list.append(img_label)
    np.save(label_path, label_list)
    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    get_cifar_bin()
