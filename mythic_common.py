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
    args = None
    debug = False
    raw_text_filename = None
    filename = None
    model = None
    epochs = None
    print_every = None
    hidden_size = None
    layers = None
    learning_rate = None
    chunk_len = None
    batch_size = None
    cuda = False

    def __init__(self):
        # Read the command line
        self.__get_command_line()
        # Add the command line info into the config dict
        self.__args_to_config()
        self.file, self.file_len = read_file(self.filename)

    def __args_to_config(self):
        """
        Takes the argparse object and puts the values into this object
        (there's probably a way better way to do this BTW)
        """
        self.debug = self.args.debug
        self.raw_text_filename = self.args.rawtext
        self.filename = self.args.filename
        self.model = self.args.model
        self.epochs = self.args.epochs
        self.print_every = self.args.print_every
        self.hidden_size = self.args.hidden_size
        self.layers = self.args.layers
        self.learning_rate = self.args.learning_rate
        self.chunk_len = self.args.chunk_len
        self.batch_size = self.args.batch_size
        self.cuda = self.args.cuda

    def __get_command_line(self):
        """
        Get command line information using the argparse module
        """
        ap = argparse.ArgumentParser(description='Does cool stuff!')
        ap.add_argument('--debug', dest='debug', action='store_true',
                        help='Switch to activate debug mode.')
        ap.set_defaults(debug=False)
        ap.add_argument('--rawtext', '-r', type=str,
                        help='Raw data file (ascii text)', required=False)

        ap.add_argument('--filename', type=str,
                        help='RAW DATA', required=False)
        ap.add_argument('--model', type=str, default="gru",
                        help='Model type', required=False)
        ap.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to run for', required=False)
        ap.add_argument('--print_every', type=int, default=100,
                        help='Print results every n epochs', required=False)
        ap.add_argument('--hidden_size', type=int, default=100,
                        help='Number of hidden layers', required=False)
        ap.add_argument('--layers', type=int, default=2,
                        help='Number of layers', required=False)
        ap.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate', required=False)
        ap.add_argument('--chunk_len', type=int, default=200,
                        help='Chunk size', required=False)
        ap.add_argument('--batch_size', type=int, default=100,
                        help='Batch size', required=False)
        ap.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Switch to activate CUDA support.')
        ap.set_defaults(cuda=False)

        # For the writer
        ap.add_argument('-s', '--init_string', type=str, default='A',
                        help='Initial seed string', required=False)
        ap.add_argument('-l', '--predict_length', type=int, default=100,
                        help='Length of the prediction', required=False)
        ap.add_argument('-t', '--temperature', type=float, default=0.8,
                        help='Temperature setting (higher is more random)', required=False)


        self.args = ap.parse_args()


def report_sys_info():
    # Report basic system stats
    log.out.info("Python version  : " + sys.version)
    log.out.info("Number of CPUs  : " + str(psutil.cpu_count()))
    log.out.info("Memory total    : " + str(round(float(psutil.virtual_memory().total) / 2 ** 30, 2)) + "GB")
    log.out.info("Memory useage   : " + str(round(float(psutil.virtual_memory().used) / 2 ** 30, 2)) + "GB")
    log.out.info("Memory available: " + str(round(float(psutil.virtual_memory().available) / 2 ** 30, 2)) + "GB")


def read_movie_info(filename, key="title", sample=None):
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
#
#     # Prepare the dataset of input to output pairs encoded as integers
#     if pattern_size is None:
#         pattern_size = int(median_overview_length / 2)
#
#     log.out.info("Training with sequence length: characters: " + str(pattern_size))
#     data_x = []
#     data_y = []
#     for i in tqdm(range(0, num_chars - pattern_size, 1)):
#         seq_in = movie_info_clean[i:i + pattern_size]
#         seq_out = movie_info_clean[i + pattern_size]
#         data_x.append([char_to_int[char] for char in seq_in])
#         data_y.append(char_to_int[seq_out])
#         num_patterns = len(data_x)
#     log.out.info("Total Patterns: " + str(num_patterns))
#
#     # Reshape for keras LSTM. 'vecs' is in the form: [samples, time steps, features]
#     vec_x = np.reshape(data_x, (num_patterns, pattern_size, 1))
#     # Normalize the vectors
#     vec_x = vec_x / float(num_vocab)
#     # One-hot encode the output variable
#     # (Encodes text into a list of word indexes in a vocabulary of size n)
#     # TODO this can blow memory on the GPU
#     # IDEA: word_to_vec gensym
#     vec_y = np_utils.to_categorical(data_y)
#     return data_x, vec_x, vec_y, sorted_chars


def read_file(filename):
    """
    Open a file and returns its handle and length
    :param filename:
    :return:
    """
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


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
