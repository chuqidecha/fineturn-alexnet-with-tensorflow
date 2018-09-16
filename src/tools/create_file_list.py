# -*- coding: utf-8 -*-
# @Time    : 18-9-12 8:48pm
# @Author  : yinwb
# @File    : create_file_list.py

import argparse
import codecs
import os

import numpy as np


def persist_file_list(output, dataset):
    with codecs.open(output, 'w', encoding='utf-8') as writer:
        for data in dataset:
            writer.write(' '.join([*data, '\n']))


def create_file_list(root_dir, output, class_dict, validation=None, test=None, seed=None):
    dataset = []
    for pathname in os.listdir(root_dir):
        full_pathname = os.path.join(root_dir, pathname)
        if os.path.isdir(full_pathname):
            filenames = os.listdir(full_pathname)
            if class_dict is not None:
                class_id = class_dict[pathname]
            else:
                class_id = pathname
            sub_dataset = [(os.path.join(pathname, filename), class_id) for filename in filenames]
            dataset.extend(sub_dataset)

    # shuffle
    if seed is not None:
        np.random.seed(seed)
    random_dataset = sorted(dataset, key=lambda x: x[0])
    np.random.shuffle(random_dataset)

    # split dataset into train,validation and test
    [pathname, filename] = os.path.split(output)
    output_pathname = root_dir if len(pathname) == 0 else os.path.join(root_dir, pathname)

    start, end = 0, 0
    num_examples = len(dataset)
    output_format = '{0}{1}'

    # validation
    if validation is not None:
        end = start + int(num_examples * validation)
        output_filename = os.path.join(output_pathname, output_format.format('validation_', filename))
        persist_file_list(output_filename, random_dataset[start:end])
        start = end

    # test
    if test is not None:
        end = start + int(num_examples * test)
        output_filename = os.path.join(output_pathname, output_format.format('test_', filename))
        persist_file_list(output_filename, random_dataset[start:end])
        start = end

    # train or all
    output_filename = os.path.join(output_pathname, output_format.format('train_' if start != 0 else '', filename))
    persist_file_list(output_filename, random_dataset[start:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="root path", type=str)
    parser.add_argument("output", help="output path related to root dir", type=str)
    parser.add_argument("--id", help="the id of every class", type=str)
    parser.add_argument("--test", help="the percentage of test", type=float)
    parser.add_argument("--validation", help="the percentage of validation,", type=float)
    parser.add_argument("--seed", help="the seed of random,", type=int)
    args = parser.parse_args()
    print(args)
    if args.id is not None:
        ids = np.loadtxt(os.path.join(args.root_dir, args.id), dtype=str)
        class_dict = dict(zip(ids[:, 1], ids[:, 0]))
    create_file_list(args.root_dir, args.output, class_dict, args.test, args.validation, args.seed)
