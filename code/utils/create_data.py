from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
import pathlib
import argparse
import datetime
import struct
import numpy as np
import pandas as pd
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


def abstract2sents(abstract, start, end):
  """Splits abstract text from datafile into list of sentences.

  Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences

  Returns:
    sents: List of sentence strings (no tags)"""
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(start, cur)
      end_p = abstract.index(end, start_p + 1)
      cur = end_p + len(end)
      sents.append(abstract[start_p+len(start):end_p])
    except ValueError as e: # no more sentences

      return  ' '.join(sents)  # string


def read_article_file(filename):
    article_text=[]
    reader = open(filename, 'rb')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        example = example_pb2.Example.FromString(example_str)
        article_text.append(example.features.feature['article'].bytes_list.value[0]) # the article text was saved under the key 'article' in the data files

    return article_text

def read_abstract_file(filename):
    abstract_text=[]
    reader = open(filename, 'rb')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        example = example_pb2.Example.FromString(example_str)
        abstract_text.append(abstract2sents(str(example.features.feature['abstract'].bytes_list.value[0]),SENTENCE_START, SENTENCE_END)) # the abstract text was saved under the key 'abstract' in the data files

    return abstract_text

def write_file(filename, data):
    with open(filename, "w") as output:
        for s in data:
            output.write(str(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process input data.')

    # default output directories:
    #   vocab file:  ../output/vocab
    #   articles:    ../output/articles/*
    #   abstracts:   ../output/abstract/*
    #
    # default input directories:
    # input: ../data/cnn-dailymail/finished_files/*
    #    chunked  test.bin  train.bin  val.bin  vocab
    #
    # and: ../data/cnn-dailymail/finished_files/chunked/*
    #    test_*, train_*, val_*

    parser.add_argument('--data_dir', metavar='dir', default='../data/cnn-dailymail/finished_files',
            help='input dir, default=../data/cnn-dailymail/finished_files/{vocab, chunked}')

    parser.add_argument('--vocab', metavar='file', default=None,
            help='input original vocab file; default=../data/cnn-dailymail/finished_files/vocab')

    parser.add_argument('--vocab_special', action='store_true', default=False,
            help='include numbers and special characters in vocab (default=False)')

    parser.add_argument('--vocab_max_num', metavar='N', type=int, default=150000,
            help='limit vocab to {N} largest values (for full vocab, use N=0) (default=150000)')

    parser.add_argument('--vocab_min_num', metavar='N', type=int, default=0,
            help='limit vocab words occuring at least {N} times (for full vocab, use N=0) (default=0)')

    parser.add_argument('--out_dir', metavar='dir', default='../output',
            help='output dir; default=../output/{vocab, articles/, abstract/}')

    args = parser.parse_args()

    input_data_dir = join(args.data_dir, 'chunked')
    input_vocab_file = args.vocab if args.vocab else join(args.data_dir, 'vocab')

    target_article_dir = join(args.out_dir, 'article')
    target_abstract_dir = join(args.out_dir, 'abstract')
    target_vocab_dir = join(args.out_dir, 'vocab')

    skip = {'abstract':False, 'article':False, 'vocab':False }

    print("==Running create_data: ", datetime.datetime.now())
    print("==Checking outputs:")

    for subdir, name in (target_abstract_dir, 'abstract'), (target_article_dir, 'article'), (target_vocab_dir, 'vocab'):
        if os.path.exists(subdir):
            print("** Warning: skipping '%s'; target directory already exists: %s" % (name, subdir))
            skip[name]=True
        else:
            print("...creating '%s' output directory: %s" %  (name, subdir))
            pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)

    print("skip? ", skip)

    print("==Checking inputs:")
    for input_file in (input_data_dir, input_vocab_file):
        found = "[ found ]" if os.path.exists(input_file) else "[missing]"
        print("\t {} \t {}".format(found, input_file))

    if not skip['vocab'] and not os.path.exists(input_vocab_file):
        raise ValueError('Input vocab data not found, exiting.')

    if not skip['article'] and not skip['abstract'] and not os.path.exists(input_vocab_file):
        raise ValueError('Input data not found, exiting.')


    print("==Continuing...")
    print("...reading from: %s" % input_data_dir)

    if skip['article'] and skip['abstract']:
        print("...skipping processing articles/abstracts; directories exist: %s" % input_data_dir)
    else:
        if not skip['article']:
            print("...processing articles to : %s" % target_article_dir)

        if not skip['abstract']:
            print("...processing abstracts to: %s" % target_abstract_dir)

        for filename in os.listdir(input_data_dir):
            if filename.startswith('val_') or filename.startswith('train_')or filename.startswith('test_') :
                if not skip['article']:
                    write_file(join(target_article_dir, filename), read_article_file(join(input_data_dir, filename)))
                if not skip['abstract']:
                    write_file(join(target_abstract_dir, filename), read_abstract_file(join(input_data_dir, filename)))

    if not skip['vocab']:
        print("...reading original vocab file (limit: %s): %s" % (args.vocab_max_num, input_vocab_file))
        df=pd.read_csv(input_vocab_file, header=None, sep=" ")
        print("...read csv: %s rows (len=%s, count=%s)" % (df.shape[0], len(df.index), df[1].count()))

        #print('...dropping NA:\n%s' % df[df.isnull().any(axis=1)])
        df.dropna(axis=0, inplace=True)
        print("...dropped NA: rows=%s" % len(df.index))

        # limit vocab to alpha-numeric words
        if not args.vocab_special:
            pattern=r'^[-:\.,0-9]*$'
            #print('=========dropping:\n%s=========\n' % df[~df[0].str.contains(pattern)])
            df = df[~df[0].str.contains(pattern)]
            print("...removed numbers: rows=%s" % len(df.index))

            pattern=r'^[^\w]*$'
            df = df[~df[0].str.contains(pattern)]
            print("...removed special: rows=%s" % len(df.index))

        # limit vocab list to words appearing more than N times
        if args.vocab_min_num:
            #print("\n====omitting:=====\n%s\n=====\n" % df[df[1] < args.vocab_min_num])
            df = df[df[1] >= args.vocab_min_num]
            print("...limit to words occuring at least %s times (rows=%s)" % (args.vocab_min_num, len(df.index)))

        df.drop_duplicates(inplace=True)
        print("...dedupe: rows=%s" % len(df.index))

        # limit vocab list to the most frequent N words
        if args.vocab_max_num:
            df = df.nlargest(args.vocab_max_num, [1])
            print("...limit to %s rows (verify: rows=%s)" % (args.vocab_max_num, len(df.index)))

        print("...writing new vocab file (rows=%s): %s/vocab" % (len(df.index), target_vocab_dir))
        df.to_csv(join(target_vocab_dir, 'vocab'), columns=[0], index=False, header=False)

    #print("=========Sample========\n%s" % df)
    print("==Done setting up data: ", datetime.datetime.now())

