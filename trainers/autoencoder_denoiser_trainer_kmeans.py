from base.base_train_multi import BaseTrainMulti
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time
from utils.evaluations import save_results,determine_normality_param,predict_anomaly
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import load 

import os
class AutoencoderDenoiserTrainer(BaseTrainMulti):


    def __init__(self, sess, model, data, config, logger):
        super(AutoencoderDenoiserTrainer, self).__init__(sess, model, data, config, logger)
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
        self.model.save(self.sess)

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
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries)
        # Check for reconstruction
        if cur_epoch % self.config.log.frequency_test == 0:
            image_eval = self.sess.run(image)
            feed_dict = {self.model.image_input: image_eval, self.model.is_training_ae: False}
            reconstruction = self.sess.run(self.model.summary_op_den, feed_dict=feed_dict)
            self.summarizer.add_tensorboard(step=cur_epoch, summaries=[reconstruction])
        den_m = np.mean(den_losses)
        self.logger.info(
            "Epoch: {} | time = {} s | loss DEN= {:4f} ".format(
                cur_epoch, time() - begin, den_m
            )
        )
        self.model.save(self.sess)

    def train_step_ae(self, image, cur_epoch):
        image_eval = self.sess.run(image)
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
        image_eval = self.sess.run(image)
        feed_dict = {
            self.model.image_input: image_eval,
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
        inference_time = []
        true_labels = []
        pipe_output = []
        pipe_delta = []
        file_writer = tf.summary.FileWriter(os.path.join(self.config.log.summary_dir, "test"))


        train_recon = []
        trainreconloop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        self.data.valid_image
        for _ in trainreconloop:
            train_batch = self.sess.run(self.data.image)
            feed_dict = {self.model.image_input: train_batch, self.model.is_training_ae: False}
            train_output_batch = self.sess.run(self.model.output, feed_dict=feed_dict).tolist()
            train_recon+=train_output_batch

        train_recon = np.reshape(np.array(train_recon),[len(train_recon),self.config.data_loader.image_size**2])
        pca = PCA(n_components = 500)
        km = KMeans(1000)
        pipeline = Pipeline([
        ('pca', pca),('cluster', km)
        ])

        pipeline.fit(train_recon)
        from sklearn.externals import joblib
        joblib.dump(pipeline, 'pipeline_k1000_pca500.pkl')
        train_codebook = pipeline.steps[1][1].cluster_centers_

        validreconloop = tqdm(range(320))
        valid_recon = []
        for _ in validreconloop:
            valid_batch = self.sess.run(self.data.valid_image)
            feed_dict = {self.model.image_input: valid_batch, self.model.is_training_ae: False}
            valid_output_batch = self.sess.run(self.model.output, feed_dict=feed_dict).tolist()
            valid_recon+=valid_output_batch


        # Create the scores
        test_loop = tqdm(range(self.config.data_loader.num_iter_per_test))
        pred_labels = []
        scores_km = []
        for cur_epoch in test_loop:
            test_batch_begin = time()
            test_batch, test_labels = self.sess.run([self.data.test_image, self.data.test_label])
            test_loop.refresh()  # to show immediately the update
            sleep(0.01)
            feed_dict = {self.model.image_input: test_batch, self.model.is_training_ae: False}
            scores_rec += self.sess.run(self.model.rec_score, feed_dict=feed_dict).tolist()
            scores_den += self.sess.run(self.model.den_score, feed_dict=feed_dict).tolist()
            scores_pipe += self.sess.run(self.model.pipe_score, feed_dict=feed_dict).tolist()
            pipe_delta_batch = self.sess.run(self.model.pipe_delta, feed_dict=feed_dict).tolist()
            pipe_delta += pipe_delta_batch
            pipe_output_batch = self.sess.run(self.model.output, feed_dict=feed_dict).tolist()
            pipe_output+=pipe_output_batch
            inference_time.append(time() - test_batch_begin)
            true_labels += test_labels.tolist()
            test_batch = pipe.step[0,1].transform(test_batch)
            pred_labels_temp ,scores_km_temp= predict_anomaly(test_batch,train_codebook,0)
            # pred_labels.append(pred_labels_temp) 
            scores_km += (scores_km_temp.tolist())
         
            
        # np.save('pred_labels',pred_labels)
        true_labels = np.asarray(true_labels)
        inference_time = np.mean(inference_time)
        self.logger.info("Testing: Mean inference time is {:4f}".format(inference_time))
        scores_rec = np.asarray(scores_rec)
        scores_den = np.asarray(scores_den)
        scores_pipe = np.asarray(scores_pipe)
        scores_km = np.asarray(scores_km)

        step = self.sess.run(self.model.global_step_tensor)
        percentiles = np.asarray(self.config.trainer.percentiles)
        save_results(
            self.config.log.result_dir,
            scores_rec,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "scores_rec",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )
        save_results(
            self.config.log.result_dir,
            scores_km,
            true_labels,
            self.config.model.name,
            self.config.data_loader.dataset_name,
            "scores_km",
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
            "scores_den",
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
            "scores_pipe",
            "paper",
            self.config.trainer.label,
            self.config.data_loader.random_seed,
            self.logger,
            step,
            percentile=percentiles,
        )


