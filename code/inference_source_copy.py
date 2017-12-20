# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding, including running ROUGE evaluation """

import os
import time
import tensorflow as tf
import data
import json
import pyrouge
import utils.misc_utils as util
import logging
import numpy as np

hps = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


def check_duplicate(input_list, n):
    n_grams = [item for item in zip(*[input_list[i:] for i in range(n)])]
    dup = 0
    for i in range(len(n_grams) - 1):
        for j in range(i + 1, len(n_grams)):
            if n_grams[j] == n_grams[i]: dup = +1

    return dup

class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        """Hypothesis constructor.

        Args:
          tokens: List of integers. The ids of the tokens that form the summary so far.
          log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
          state: Current state of the decoder, a LSTMStateTuple.
          attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
          p_gens: List, same length as tokens, of floats, or None if not using source-copy model. The values of the generation probability so far.
          coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
          token: Integer. Latest token produced by beam search.
          log_prob: Float. Log prob of the latest token.
          state: Current decoder state, a LSTMStateTuple.
          attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
          p_gen: Generation probability on latest step. Float.
          coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
          New Hypothesis for next step.
        """
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        weight = 100000 * check_duplicate(self.tokens, 3) + 1
        # Reduce the probability of any hypothesis with duplicate trigram

        return sum(self.log_probs) * weight

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch):
    """Performs beam search decoding on the given example.

    Args:
      sess: a tf.Session
      model: a seq2seq model
      vocab: Vocabulary object
      batch: Batch object that is the same example repeated across the batch

    Returns:
      best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """
    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_batch.shape[1]])  # zero vector of length attention_length
                       ) for _ in range(hps.beam_size)]
    results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)

    steps = 0
    while steps < hps.max_dec_steps and len(results) < hps.beam_size:
        latest_tokens = [h.latest_token for h in hyps]  # latest token produced by each hypothesis
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in
                         latest_tokens]  # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
        states = [h.state for h in hyps]  # list of current decoder states of the hypotheses
        prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)

        # Run one step of the decoder to get the new info
        (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                        batch=batch,
                                                                                                        latest_tokens=latest_tokens,
                                                                                                        enc_states=enc_states,
                                                                                                        dec_init_states=states,
                                                                                                        prev_coverage=prev_coverage)

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(
            hyps)  # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
        for i in range(num_orig_hyps):
            h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                             new_coverage[
                                                                 i]  # take the ith hypothesis and new decoder state info
            for j in range(hps.beam_size * 2):  # for each of the top 2*beam_size hyps:
                # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = []  # will contain hypotheses for the next step
        for h in sort_hyps(all_hyps):  # in order of most likely h
            if h.latest_token == vocab.word2id(data.STOP_DECODING):  # if stop token is reached...
                # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= hps.min_dec_steps:
                    results.append(h)
            else:  # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)
            if len(hyps) == hps.beam_size or len(results) == hps.beam_size:
                # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(
            results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted[0]


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher, vocab):
        """Initialize decoder.

        Args:
          model: a Seq2SeqAttentionModel object.
          batcher: a Batcher object.
          vocab: Vocabulary object
        """
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = tf.train.Saver()  # we use this to load checkpoints for decoding
        self._sess = tf.Session(config=util.get_config())

        # Load an initial checkpoint to use for decoding
        ckpt_path = util.load_ckpt(self._saver, self._sess)

        if hps.single_pass:
            # Make a descriptive decode directory name
            ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]  # this is something of the form "ckpt-123456"
            self._decode_dir = os.path.join(hps.log_root, get_decode_dir_name(ckpt_name))
            if os.path.exists(self._decode_dir):
                raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

        else:  # Generic decode dir name
            self._decode_dir = os.path.join(hps.log_root, "decode")

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

        if hps.single_pass:
            # Make the dirs to contain output written in the correct format for pyrouge
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
        if hps.rouge_eval_only:
            if not hps.eval_path:
                raise Exception("Must specify path to folder containing decoded files for evaluation")
            else:
                self._rouge_dec_dir = hps.eval_path
                if not os.path.exists(self._rouge_dec_dir):
                    raise Exception("Folder containing decoded files for evaluation does not exist!")


    def decode(self):
        """Decode examples until data is exhausted (if hps.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        if (hps.rouge_eval_only):
            tf.logging.info("ROUGE only mode, Starting ROUGE eval...", self._rouge_ref_dir,
                            self._rouge_dec_dir)
            results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
            rouge_log(results_dict, self._decode_dir)
            return

        t0 = time.time()
        counter = 0
        while True:
            batch = self._batcher.next_batch()  # 1 example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                assert hps.single_pass, "Dataset exhausted, but we are not in single_pass mode"
                tf.logging.info("Decoder has finished reading dataset for single_pass.")
                tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...", self._rouge_ref_dir,
                                self._rouge_dec_dir)
                results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

            article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                   (batch.art_oovs[0] if hps.copy_source else None))  # string

            # Run beam search to get best Hypothesis
            best_hyp = run_beam_search(self._sess, self._model, self._vocab, batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self._vocab,
                                                 (batch.art_oovs[0] if hps.copy_source else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_output = ' '.join(decoded_words)  # single string

            if hps.single_pass:
                self.write_for_rouge(original_abstract_sents, decoded_words,
                                     counter)  # write ref summary and decoded summary to file, to eval with pyrouge later
                counter += 1  # this is how many examples we've decoded
            else:
                print_results(article_withunks, abstract_withunks, decoded_output)  # log output to screen
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists,
                                       best_hyp.p_gens)  # write info to .json file for visualization tool

                # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
                t1 = time.time()
                if t1 - t0 > SECS_UNTIL_NEW_CKPT:
                    tf.logging.info(
                        'We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint',
                        t1 - t0)
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()

    def write_for_rouge(self, reference_sents, decoded_words, ex_index):
        """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

        Args:
          reference_sents: list of strings
          decoded_words: list of strings
          ex_index: int, the index with which to label the files
        """
        # First, divide decoded output into sentences
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        # Write to file
        ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
        decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)

        with open(ref_file, "w") as f:
            for idx, sent in enumerate(reference_sents):
                f.write(str(sent.encode('utf8'))) if idx == len(reference_sents) - 1 else f.write(
                    str(sent.encode('utf8') + b"\n"))
        with open(decoded_file, "w") as f:
            for idx, sent in enumerate(decoded_sents):
                f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

        tf.logging.info("Wrote example %i to file" % ex_index)

    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        """Write some data to json file, which can be read into the in-browser attention visualizer tool:


        Args:
          article: The original article string.
          abstract: The human (correct) abstract string.
          attn_dists: List of arrays; the attention distributions.
          decoded_words: List of strings; the words of the generated summary.
          p_gens: List of scalars; the p_gen values. If not running in source-copy mode, list of None.
        """
        article_lst = article.split()  # list of words
        decoded_lst = decoded_words  # list of decoded words
        to_write = {
            'article_lst': [make_html_safe(t) for t in article_lst],
            'decoded_lst': [make_html_safe(t) for t in decoded_lst],
            'abstract_str': make_html_safe(abstract),
            'attn_dists': attn_dists
        }
        if hps.copy_source:
            to_write['p_gens'] = p_gens
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)
        tf.logging.info('Wrote visualization data to %s', output_fname)


def print_results(article, abstract, decoded_output):
    """Prints the article, the reference summmary and the decoded summary to screen"""
    print("---------------------------------------------------------------------------")
    tf.logging.info('Original Article:  %s', article)
    tf.logging.info('Reference Summary: %s', abstract)
    tf.logging.info('Model Gen Summary: %s', decoded_output)
    print("---------------------------------------------------------------------------")


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
      results_dict: the dictionary returned by pyrouge
      dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    tf.logging.info(log_str)  # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    tf.logging.info("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w") as f:
        f.write(log_str)


def get_decode_dir_name(ckpt_name):
    """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

    if "train" in hps.data_path:
        dataset = "train"
    elif "val" in hps.data_path:
        dataset = "val"
    elif "test" in hps.data_path:
        dataset = "test"
    else:
        raise ValueError("hps.data_path %s should contain one of train, val or test" % (hps.data_path))
    dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (
        dataset, hps.max_enc_steps, hps.beam_size, hps.min_dec_steps, hps.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
