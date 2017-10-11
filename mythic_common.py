""""
This module contains shared functions that do cool things.

@author : Brad Beechler (brad.e.beechler@gmail.com)
Modified: 20171234 (Brad Beechler)
"""

from uplog import log
import sys
import psutil
import argparse
from string import printable
import json
import numpy as np
import unidecode
import string
import time
import math
import torch
import random


model_file_name_pattern = "mythic_model-EE-WWWWWW_XS1_XS2_YS1_LS1"
all_characters = string.printable
num_characters = len(all_characters)


class MythicSettings:
    # For trainer
    args = None
    debug = False
    text_file = None
    model = None
    epochs = None
    print_every = None
    hidden_size = None
    layers = None
    learning_rate = None
    chunk_len = None
    batch_size = None
    cuda = False
    # For writer
    model_file = None
    seed_string = None
    predict_length = None
    temperature = False

    def __init__(self):
        # Read the command line
        self.__get_command_line()
        # Add the command line info into the config dict
        self.__args_to_config()
        if self.text_file is not None:
            self.text_string, self.text_length = read_file_as_string(self.text_file)

    def __args_to_config(self):
        """
        Takes the argparse object and puts the values into this object
        (there's probably a way better way to do this BTW)
        """
        # General
        self.debug = self.args.debug
        self.cuda = self.args.cuda
        self.model_file = self.args.model_file

        # For trainer
        self.text_file = self.args.text_file
        self.model = self.args.model
        self.epochs = self.args.epochs
        self.print_every = self.args.print_every
        self.hidden_size = self.args.hidden_size
        self.layers = self.args.layers
        self.learning_rate = self.args.learning_rate
        self.chunk_size = self.args.chunk_size
        self.batch_size = self.args.batch_size
        # For writer
        self.seed_string = self.args.seed_string
        self.predict_length = self.args.predict_length
        self.temperature = self.args.temperature


    def __get_command_line(self):
        """
        Get command line information using the argparse module
        """
        # General
        ap = argparse.ArgumentParser(description='Does cool stuff!')
        ap.add_argument('--debug', dest='debug', action='store_true',
                        help='Switch to activate debug mode.')
        ap.set_defaults(debug=False)
        ap.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Switch to activate CUDA support.')
        ap.set_defaults(cuda=False)
        ap.add_argument('--model_file', type=str, default=None,
                        help='Torch model filename (foo.pt)', required=False)
        # For the trainer
        ap.add_argument('--text_file', type=str, default=None,
                        help='TRAIN: Raw data file (ascii text)', required=False)
        ap.add_argument('--model', type=str, default="gru",
                        help='TRAIN: Model type', required=False)
        ap.add_argument('--epochs', type=int, default=2000,
                        help='TRAIN: Number of epochs to run for', required=False)
        ap.add_argument('--print_every', type=int, default=100,
                        help='TRAIN: Print results every n epochs', required=False)
        ap.add_argument('--hidden_size', type=int, default=100,
                        help='TRAIN: Number of hidden layers', required=False)
        ap.add_argument('--layers', type=int, default=2,
                        help='TRAIN: Number of layers', required=False)
        ap.add_argument('--learning_rate', type=float, default=0.01,
                        help='TRAIN: The learning rate', required=False)
        ap.add_argument('--chunk_size', type=int, default=200,
                        help='TRAIN: Chunk size', required=False)
        ap.add_argument('--batch_size', type=int, default=100,
                        help='TRAIN: Batch size', required=False)
        # For the writer
        ap.add_argument('--seed_string', type=str, default='A',
                        help='WRITE: Initial seed string', required=False)
        ap.add_argument('--predict_length', type=int, default=100,
                        help='WRITE: Length of the prediction', required=False)
        ap.add_argument('--temperature', type=float, default=0.8,
                        help='WRITE: Temperature setting (higher is more random)', required=False)

        self.args = ap.parse_args()


def report_sys_info():
    # Report basic system stats
    log.out.info("Python version  : " + sys.version)
    log.out.info("Number of CPUs  : " + str(psutil.cpu_count()))
    log.out.info("Memory total    : " + str(round(float(psutil.virtual_memory().total) / 2 ** 30, 2)) + "GB")
    log.out.info("Memory useage   : " + str(round(float(psutil.virtual_memory().used) / 2 ** 30, 2)) + "GB")
    log.out.info("Memory available: " + str(round(float(psutil.virtual_memory().available) / 2 ** 30, 2)) + "GB")


def read_movie_json(filename, key="title", sample=None):
    log.out.info("Opening: " + filename)
    json_file_handle = open(filename)
    movie_data = json.load(json_file_handle)
    log.out.info("Found: " + str(len(movie_data)) + " records.")
    movie_ids = list(movie_data.keys())
    log.out.info("Parsing records.")
    return_array = []
    if sample is not None:
        # Get a random subset
        for i in range(sample):
            rando = movie_ids[random.randint(0, len(movie_data)-1)]
            return_array.append(movie_data[rando][key])
    else:
        # Grab em all!
        for movie_id in movie_ids:
            if len(movie_data[movie_id][key]) > 0:
                return_array.append(movie_data[movie_id][key])
    return np.asarray(return_array)


def get_median_length(measure_array):
    length_function = lambda x: len(x)
    vector_function = np.vectorize(length_function)
    return int(np.median(vector_function(measure_array)))


def munge_huge_clump(text_array, lower_case=True):
    log.out.info("Munging records.")
    if lower_case:
        return ' '.join(text_array).lower()
    else:
        return ' '.join(text_array)


def clean_string_to_printable(string_in, lower=True):
    log.out.info("Size in: " + str(len(string_in)))
    if lower:
        char_filter = printable.lower()
    else:
        char_filter = printable
    string_out = "".join(c for c in string_in if c in char_filter)
    log.out.info("Size out: " + str(len(string_out)))
    return string_out


def read_file_as_string(filename):
    """
    Open a file and returns its handle and length
    :param filename:
    :return:
    """
    text_str = unidecode.unidecode(open(filename).read())
    return text_str, len(text_str)


def char_tensor(input_string):
    """
    Transform a string into a tensor
    :param input_string: the string you want to transform
    :return: tensor for torch
    """
    tensor = torch.zeros(len(input_string)).long()
    for c in range(len(input_string)):
        try:
            tensor[c] = all_characters.index(input_string[c])
        except:
            continue
    return tensor


def time_since(start_time):
    """
    :param start_time: start time
    :return: A human readable elapsed time
    """
    s = time.time() - start_time
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# def prepare_nonsense(movie_overview_array, pattern_size=None):
#     movie_info_raw = munge_huge_clump(movie_overview_array)
#     # Create mapping of printable chars to integers
#     sorted_chars = sorted(list(set(printable.lower())))
#     log.out.info("Cleaning data of weird characters.")
#     movie_info_clean = clean_string_to_printable(movie_info_raw)
#     median_overview_length = get_median_length(movie_overview_array)
#     log.out.info("Median overview length: " + str(median_overview_length))
#     char_to_int = dict((c, i) for i, c in enumerate(sorted_chars))
#     num_chars = len(movie_info_clean)
#     num_vocab = len(sorted_chars)
#     log.out.info("Total characters: " + str(num_chars) +
#                  " Total vocabulary: " + str(num_vocab))