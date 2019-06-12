from base.base_train_multi import BaseTrainMulti
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import save_results


class DAEDenoiserTrainer(BaseTrainMulti):


    def __init__(self, sess, model, data, config, logger):
        super(DAEDenoiserTrainer, self).__init__(sess, model, data, config, logger)
        self.batch_size = self.config.data_loader.batch_size
        self.noise_dim = self.config.trainer.noise_dim
        self.img_dims = self.config.trainer.image_dims
        # Inititalize the train Dataset Iterator
        self.sess.run(self.data.iterator.initializer)
        # Initialize the test Dataset Iterator
        self.sess.run(self.data.test_iterator.initializer)
        if self.config.data_loader.validation:
            self.sess.run(self.data.valid_iterator.initializer)
            self.best_valid_loss = 0
            self.nb_without_improvements = 0

    def train_epoch_ae(self):
        # Attach the epoch loop to a variable
        begin = time()
        # Make the loop of the epoch iterations
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        ae_losses = []
        summaries = []
        image = self.data.image
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            ae, sum_ae = self.train_step_ae(image, cur_epoch)
            ae_losses.append(ae)
            summaries.append(sum_ae)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries)
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            image_eval = self.sess.run(image)
            feed_dict = {self.model.image_input: image_eval, self.model.is_training_ae: False}
            reconstruction = self.sess.run(self.model.summary_op_ae, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        ae_m = np.mean(ae_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss AE= {:4f} ".format(
                cur_epoch, time() - begin, ae_m
            )
        )

    def train_epoch_den(self):
        # Attach the epoch loop to a variable
        begin = time()
        # Make the loop of the epoch iterations
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        den_losses = []
        summaries = []
        image = self.data.image
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            den, sum_den = self.train_step_den(image, cur_epoch)
            den_losses.append(den)
            summaries.append(sum_den)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries,summarizer="valid")
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            image_eval = self.sess.run(image)
            noise = np.zeros_like(image_eval)
            feed_dict = {self.model.image_input: image_eval,
            self.model.noise_tensor: noise, self.model.is_training_ae: False}
            reconstruction = self.sess.run(self.model.summary_op_den, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction],summarizer="valid")
        den_m = np.mean(den_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss DEN= {:4f} ".format(
                cur_epoch, time() - begin, den_m
            )
        )

    def train_step_ae(self, image, cur_epoch):
        noise = np.random.normal(
            loc = 0.0,
            scale = 1.0,
            size = [self.config.data_loader.batch_size] + self.config.trainer.image_dims)
        image_eval = self.sess.run(image)
        image_eval += noise
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.is_training_ae: True,
        }
        # Train Autoencoder
        _, lae, sm_ae = self.sess.run(
            [self.model.train_auto_op, self.model.auto_loss, self.model.summary_op_loss_ae],
            feed_dict=feed_dict,
        )
        return lae, sm_ae


    def train_step_den(self, image, cur_epoch):
        noise = np.random.normal(
            loc = 0.0,
            scale = 1.0,
            size = [self.config.data_loader.batch_size] + self.config.trainer.image_dims)
        image_eval = self.sess.run(image)
        feed_dict = {
            self.model.image_input: image_eval,
            self.model.noise_tensor: noise,
            self.model.is_training_ae: False,
        }
        # Train Denoiser
        _, lden, sm_den = self.sess.run(
            [self.model.train_den_op, self.model.den_loss, self.model.summary_op_loss_den],
            feed_dict=feed_dict,
        )
        return lden, sm_den

    def test_epoch(self):
        self.logger.warn("Testing evaluation...")
        scores_rec = []
        scores_den = []
        scores_pipe = []
        scores_mask1 = []
        scores_mask2 = []
        inference_time = []
        true_labels = []
        # Create the scores
        test_loop = tqdm(range(self.config.data_loader.num_iter_per_test))
        for _ in test_loop:
            test_batch_begin = time()
            test_batch, test_labels = self.sess.run([self.data.test_image, self.data.test_label])
            test_loop.refresh()  # to show immediately the update
            sleep(0.01)
            feed_dict = {self.model.image_input: test_batch, self.model.is_training_ae: False}
            scores_rec += self.sess.run(self.model.rec_score, feed_dict=feed_dict).tolist()
            scores_den += self.sess.run(self.model.den_score, feed_dict=feed_dict).tolist()
            scores_pipe += self.sess.run(self.model.pipe_score, feed_dict=feed_dict).tolist()
            scores_mask1 += self.sess.run(self.model.mask_score_1, feed_dict=feed_dict).tolist()
            scores_mask2 += self.sess.run(self.model.mask_score_2, feed_dict=feed_dict).tolist()
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        scores_rec = np.asarray(scores_rec)
        scores_den = np.asarray(scores_den)
        scores_pipe = np.asarray(scores_pipe)
        scores_mask1 = np.asarray(scores_mask1)
        scores_mask2 = np.asarray(scores_mask2)
        # scores_scaled = (scores - min(scores)) / (max(scores) - min(scores))
        step = self.sess.run(self.model.global_step_tensor)
        percentiles = np.asarray(self.config.trainer.percentiles)
        save_results(
            self.config.log.result_dir,
            scores_rec,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "rec",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_den,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "den",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_pipe,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "pipe",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_mask1,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "mask_den",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_mask2,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "mask_ae",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )


