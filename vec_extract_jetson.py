#
# file: vec_extract_jetson.py
# desc: Image Vector Extraction on Jetson Devices
# auth: Mark Swaringen
#
# copyright: Saivvy 2021
#############################################

import argparse
import sys
import os
import time

from PIL import Image
import numpy as np
import pandas as pd


def parse_arguments(args):
    """
    Method that will parse the given arguments and correctly process them
    :param args:
    :return:
    """

    # parse incoming information
    parser = argparse.ArgumentParser(
        description='Test file for testing on Jetson devices. Data path is required.'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='absolute path to the data directory'
    )
    parser.add_argument(
        '--cuda',
        type=bool,
        default=None,
        help='True to run on GPU'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='resnet model'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default=None,
        help='model layer'
    )
    parser.add_argument(
        '--vec_length',
        type=str,
        default=None,
        help='vector length for model'
    )
    args = parser.parse_args(args)

    return args

def img_2_vec(files,input_path,cuda,model_name,layer,vec_length):
    os.system('git clone "https://github.com/christiansafka/img2vec.git"')
    sys.path.append("img2vec/img2vec_pytorch")
    from img_to_vec import Img2Vec

    img2vec = Img2Vec(model=model_name,cuda=cuda,layer=layer, layer_output_size=vec_length)
    img2vec = Img2Vec(cuda=cuda)
    # vec_length = 512  # Using resnet-18 as default
    samples = 670  # Amount of samples to take from input path

    vec_mat = np.zeros((samples, vec_length))
    sample_indices = np.random.choice(range(0, len(files)), size=samples, replace=False)

    print('Reading images...')
    df = pd.DataFrame()
    for index, i in enumerate(sample_indices):
        file = files[i]
        filename = os.fsdecode(file)
        img = Image.open(os.path.join(input_path, filename))
        # img = img.to(device)
        vec = img2vec.get_vec(img)
        vec_mat[index, :] = vec
        new_row = {'filename':filename, 'sample_indices':i}
        df = df.append(new_row, ignore_index=True)

    df['sample_indices'] = df['sample_indices'].astype(int)

    return df,vec_mat

def main(args):

    args = parse_arguments(args)

    # input_path = "data/minneapple/train/images"
    # data_path = "data/minneapple/"
    data_path = args.data_path
    cuda = args.cuda
    model_name = args.model
    layer = args.layer
    vec_length = args.vec_length

    # data_path = "data/minneapple/train/images"
    input_path = data_path + "train/images"
    files = os.listdir(input_path)

    t0 = time.time()
    df,vec_mat = img_2_vec(files,input_path,cuda,model_name,layer,vec_length)
    t1 = time.time() - t0
    print('Vectors created!')
    print("Time: ", t1,"seconds")
    print("Embeddings extracted: ", len(files))
    print(model_name, layer)

    # df.to_csv(data_path + 'vectors/res18_vector_matrix_train_filenames.csv',index=False)
    # np.save(data_path + 'vectors/res18_vector_matrix_train.npy', vec_mat)
    # print('Vectors saved!')

if __name__ == "__main__":
    main(sys.argv[1:])