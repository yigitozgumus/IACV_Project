import tensorflow as tf
import os


class Summarizer:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.log.summary_dir, "train_ae"), self.sess.graph
        )
        self.valid_summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.log.summary_dir, "train_den"), self.sess.graph
        )
        self.test_summary_writer = tf.summary.FileWriter(
            os.path.join(self.config.log.summary_dir, "test"), self.sess.graph
        )

    # it can summarize scalars and images.
    def add_tensorboard(self, step, summarizer="train_ae", scope="", summaries=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """

        if summarizer  == "train_ae":
            summary_writer = self.train_summary_writer
        elif summarizer == "train_den":
            summary_writer = self.valid_summary_writer
        elif summarizer == "test":
            summary_writer = self.test_summary_writer

        with tf.variable_scope(scope):
            for summary in summaries:
                summary_writer.add_summary(summary, step)
            summary_writer.flush()
