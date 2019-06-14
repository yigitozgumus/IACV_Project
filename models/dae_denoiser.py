from base.base_model import BaseModel
import tensorflow as tf
from utils.utils import get_getter

class DAEDenoiser(BaseModel):
    def __init__(self, config):
        super(DAEDenoiser, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders
        self.is_training_ae = tf.placeholder(tf.bool)
        #self.is_training_den = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.ground_truth = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="gt"
        )
        self.noise_tensor = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="noise"
        )

        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        ## Architecture

        # Encoder Decoder Part first
        self.logger.info("Building Training Graph")
        with tf.variable_scope("DAE_Denoiser"):
            self.noise_gen, self.rec_image = self.autoencoder(self.image_input)
            self.output, self.mask = self.denoiser(self.rec_image + self.noise_tensor)

        # Loss Function
        with tf.name_scope("Loss_Function"):
            with tf.name_scope("Autoencoder"):
                # Contextual Loss
                delta_enc = self.rec_image - self.image_input
                delta_enc = tf.layers.Flatten()(delta_enc)
                self.auto_loss = tf.reduce_mean(
                    tf.norm(
                        delta_enc,
                        ord=self.config.trainer.enc_norm_degree,
                        axis=1,
                        keepdims=False,
                    )
                )
            with tf.name_scope("Denoiser"):
                delta_den = self.output - self.image_input
                delta_den = tf.layers.Flatten()(delta_den)
                self.den_loss = tf.reduce_mean(
                    tf.norm(
                        delta_den,
                        ord=self.config.trainer.den_norm_degree,
                        axis=1,
                        keepdims=False,
                    )
                )

        # Optimizer
        with tf.name_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                self.config.trainer.l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            # Collect All Variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.autoencoder_vars = [
                v
                for v in all_variables
                if v.name.startswith("DAE_Denoiser/Autoencoder")
            ]
            self.denoiser_vars = [
                v
                for v in all_variables
                if v.name.startswith("DAE_Denoiser/Denoiser")
            ]
            self.auto_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="DAE_Denoiser/Autoencoder"
            )
            self.den_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="DAE_Denoiser/Denoiser"
            )
            with tf.control_dependencies(self.auto_update_ops):
               self.auto_op = self.optimizer.minimize(
                    self.auto_loss,
                    var_list=self.autoencoder_vars,
                    global_step=self.global_step_tensor,
                )

            with tf.control_dependencies(self.den_update_ops):
                self.den_op = self.optimizer.minimize(self.den_loss, var_list=self.denoiser_vars)

            # Exponential Moving Average for Estimation
            self.auto_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_auto = self.auto_ema.apply(self.autoencoder_vars)

            self.den_ema = tf.train.ExponentialMovingAverage(decay=self.config.trainer.ema_decay)
            maintain_averages_op_den = self.den_ema.apply(self.denoiser_vars)

            with tf.control_dependencies([self.auto_op]):
                self.train_auto_op = tf.group(maintain_averages_op_auto)

            with tf.control_dependencies([self.den_op]):
                self.train_den_op = tf.group(maintain_averages_op_den)

        self.logger.info("Building Testing Graph...")
        with tf.variable_scope("DAE_Denoiser"):
            self.noise_gen_ema, self.rec_image_ema = self.autoencoder(self.image_input ,getter=get_getter(self.auto_ema))
            self.output_ema, self.mask_ema = self.denoiser(self.rec_image_ema,getter=get_getter(self.den_ema))
            self.residual = self.input_image - self.mask_ema

        with tf.name_scope("Testing"):
            with tf.variable_scope("Reconstruction_Loss"):
                # |x - D(E(x)) |2
                delta = self.rec_image_ema - self.image_input
                delta = tf.layers.Flatten()(delta)
                self.rec_score = tf.norm(delta, ord=2, axis=1, keepdims=False)
            with tf.variable_scope("Denoising_Loss"):
                delta_den = self.output_ema - self.rec_image_ema
                delta_den = tf.layers.Flatten()(delta_den)
                self.den_score = tf.norm(delta_den, ord=2, axis=1,keepdims=False)
            with tf.variable_scope("Pipeline_Loss_1"):
                delta_pipe = self.output_ema - self.image_input
                delta_pipe = tf.layers.Flatten()(delta_pipe)
                self.pipe_score = tf.norm(delta_pipe, ord=1,axis=1,keepdims=False)
            with tf.variable_scope("Pipeline_Loss_2"):
                delta_pipe = self.output_ema - self.image_input
                delta_pipe = tf.layers.Flatten()(delta_pipe)
                self.pipe_score_2 = tf.norm(delta_pipe, ord=2,axis=1,keepdims=False)
            with tf.variable_scope("Mask_1"):
                delta_mask = (self.input_image - self.mask_ema) 
                delta_mask = tf.layers.Flatten()(delta_mask)
                self.mask_score_1 = tf.norm(delta_mask, ord=1,axis=1,keepdims=False)
            with tf.variable_scope("Mask_2"):
                delta_mask_2 = (self.input_image - self.mask_ema) 
                delta_mask_2 = tf.layers.Flatten()(delta_mask_2)
                self.mask_score_2 = tf.norm(delta_mask_2, ord=2,axis=1,keepdims=False)

        # Summary
        with tf.name_scope("Summary"):
            with tf.name_scope("autoencoder_loss"):
                tf.summary.scalar("loss_auto", self.auto_loss, ["loss_ae"])
            with tf.name_scope("denoiser_loss"):
                tf.summary.scalar("loss_den", self.den_loss, ["loss_den"])
            with tf.name_scope("Image"):
                tf.summary.image("Input_Image", self.image_input, 1, ["image"])
                tf.summary.image("rec_image",self.rec_image,1, ["image"])
                tf.summary.image("mask", self.mask, 1, ["image_2"])
                tf.summary.image("Output_Image", self.output, 1, ["image_2"])
                tf.summary.image("Rec_Image", self.rec_image, 1, ["image_2"])
                tf.summary.image("Input_Image", self.image_input, 1, ["image_2"])
                tf.summary.image("mask", self.mask, 1, ["image_3"])
                tf.summary.image("Output_Image", self.output, 1, ["image_3"])
                tf.summary.image("Rec_Image", self.rec_image, 1, ["image_3"])
                tf.summary.image("Input_Image", self.image_input, 1, ["image_3"])
                tf.summary.image("Residual", self.residual,1,["image_3"])
                tf.summary.image("Ground_Truth", self.ground_truth,1,["image_3"])

        self.summary_op_ae = tf.summary.merge_all("image")
        self.summary_op_den = tf.summary.merge_all("image_2")
        self.summary_op_test = tf.summary.merge_all("image_3")
        self.summary_op_loss_ae = tf.summary.merge_all("loss_ae")
        self.summary_op_loss_den = tf.summary.merge_all("loss_den")
        self.summary_all_ae = tf.summary.merge([self.summary_op_ae, self.summary_op_loss_ae])
        self.summary_all_den = tf.summary.merge([self.summary_op_den, self.summary_op_loss_den])
        #self.summary_all = tf.summary.merge([self.summary_op_im, self.summary_op_loss])

    def autoencoder(self, image_input, getter=None):
        # This generator will take the image from the input dataset, and first it will
        # it will create a latent representation of that image then with the decoder part,
        # it will reconstruct the image.
        with tf.variable_scope(
            "Autoencoder", custom_getter=getter, reuse=tf.AUTO_REUSE
        ):
            with tf.variable_scope("Encoder"):
                x_e = tf.reshape(
                    image_input,
                    [
                        -1,
                        self.config.data_loader.image_size,
                        self.config.data_loader.image_size,
                        1,
                    ],
                )
                net_name = "Layer_1"
                with tf.variable_scope(net_name):
                    x_e = tf.layers.Conv2D(
                        filters=128,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )(x_e)
                    x_e = tf.nn.leaky_relu(
                        features=x_e,
                        alpha=self.config.trainer.leakyReLU_alpha,
                        name="leaky_relu",
                    )
                    # 14 x 14 x 64
                net_name = "Layer_2"
                with tf.variable_scope(net_name):
                    x_e = tf.layers.Conv2D(
                        filters=256,
                        kernel_size=5,
                        padding="same",
                        strides=(2, 2),
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )(x_e)
                    x_e = tf.layers.batch_normalization(
                        x_e,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training_ae,
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e,
                        alpha=self.config.trainer.leakyReLU_alpha,
                        name="leaky_relu",
                    )
                    # 7 x 7 x 128
                net_name = "Layer_3"
                with tf.variable_scope(net_name):
                    x_e = tf.layers.Conv2D(
                        filters=512,
                        kernel_size=5,
                        padding="same",
                        strides=(2, 2),
                        kernel_initializer=self.init_kernel,
                        name="conv",
                    )(x_e)
                    x_e = tf.layers.batch_normalization(
                        x_e,
                        momentum=self.config.trainer.batch_momentum,
                        epsilon=self.config.trainer.batch_epsilon,
                        training=self.is_training_ae,
                    )
                    x_e = tf.nn.leaky_relu(
                        features=x_e,
                        alpha=self.config.trainer.leakyReLU_alpha,
                        name="leaky_relu",
                    )
                    # 4 x 4 x 256
                x_e = tf.layers.Flatten()(x_e)
                net_name = "Layer_4"
                with tf.variable_scope(net_name):
                    x_e = tf.layers.Dense(
                        units=self.config.trainer.noise_dim,
                        kernel_initializer=self.init_kernel,
                        name="fc",
                    )(x_e)

            noise_gen = x_e

            with tf.variable_scope("Decoder"):
                net = tf.reshape(noise_gen, [-1, 1, 1, self.config.trainer.noise_dim])
                net_name = "layer_1"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=512,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv1",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_ae,
                        name="tconv1/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv1/relu")

                net_name = "layer_2"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=256,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv2",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_ae,
                        name="tconv2/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv2/relu")

                net_name = "layer_3"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=128,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv3",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_ae,
                        name="tconv3/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv3/relu")
                net_name = "layer_4"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv4",
                    )(net)
                    net = tf.layers.batch_normalization(
                        inputs=net,
                        momentum=self.config.trainer.batch_momentum,
                        training=self.is_training_ae,
                        name="tconv4/bn",
                    )
                    net = tf.nn.relu(features=net, name="tconv3/relu")

                net_name = "layer_5"
                with tf.variable_scope(net_name):
                    net = tf.layers.Conv2DTranspose(
                        filters=1,
                        kernel_size=5,
                        strides=(2, 2),
                        padding="same",
                        kernel_initializer=self.init_kernel,
                        name="tconv5",
                    )(net)
                    net = tf.nn.tanh(net, name="tconv5/tanh")

            image_rec = net

            return noise_gen, image_rec

    def denoiser(self, image_input, getter=None):
        # Full Model Scope
        with tf.variable_scope("Denoiser", reuse=tf.AUTO_REUSE, custom_getter=getter):
            # First Convolution + ReLU layer
            net = tf.layers.Conv2D(
                filters=63,
                kernel_size=3,
                strides=1,
                kernel_initializer=self.init_kernel,
                padding="same",
            )(image_input)
            net = tf.nn.relu(features=net)
            # 1 Convolution of the image for the bottom layer
            net_input = tf.layers.Conv2D(
                filters=1,
                kernel_size=3,
                strides=1,
                kernel_initializer=self.init_kernel,
                padding="same",
            )(image_input)
            net_layer_1 = tf.layers.Conv2D(
                filters=1,
                kernel_size=3,
                strides=1,
                kernel_initializer=self.init_kernel,
                padding="same",
            )(net)
            # First convolution from the image second one from the first top layer convolution
            mask = net_input + net_layer_1

            for i in range(19):
                # Top layer chained convolutions
                net = tf.layers.Conv2D(
                    filters=63,
                    kernel_size=3,
                    strides=1,
                    kernel_initializer=self.init_kernel,
                    padding="same",
                )(net)
                net = tf.nn.relu(features=net)
                # Bottom layer single convolutions
                net_1 = tf.layers.Conv2D(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    kernel_initializer=self.init_kernel,
                    padding="same",
                )(net)
                mask += net_1
            output = image_input + mask

        return output, mask

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
