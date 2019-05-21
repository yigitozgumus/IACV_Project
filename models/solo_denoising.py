from base.base_model import BaseModel
import tensorflow as tf


class Denoising(BaseModel):
    def __init__(self, config):
        super(Denoising, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders
        self.is_training = tf.placeholder(tf.bool)
        self.image_input = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.label = tf.placeholder(
            tf.float32, shape=[None] + self.config.trainer.image_dims, name="x"
        )
        self.init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)

        # Full Model Scope
        with tf.variable_scope("Denoising", reuse=tf.AUTO_REUSE):

            # First Convolution + ReLU layer
            net = tf.layers.Conv2D(
                filters=63,
                kernel_size=3,
                strides=1,
                kernel_initializer=self.init_kernel,
                padding="same",
            )(self.image_input)
            net = tf.nn.relu(features=net)
            # 1 Convolution of the image
            net_input = tf.layers.Conv2D(
                filters=1,
                kernel_size=3,
                strides=1,
                kernel_initializer=self.init_kernel,
                padding="same",
            )(self.image_input)
            net_layer_1 = tf.layers.Conv2D(
                filters=1,
                kernel_size=3,
                strides=1,
                kernel_initializer=self.init_kernel,
                padding="same",
            )(net)
            # Add to the image
            self.output = net_input + net_layer_1

            for i in range(19):
                net = tf.layers.Conv2D(
                    filters=63,
                    kernel_size=3,
                    strides=1,
                    kernel_initializer=self.init_kernel,
                    padding="same",
                )(net)
                net = tf.nn.relu(features=net)
                net_1 = tf.layers.Conv2D(
                    filters=1,
                    kernel_size=3,
                    strides=1,
                    kernel_initializer=self.init_kernel,
                    padding="same",
                )(net)
                self.output += net_1
            self.output += self.image_input
        # Loss Function
        with tf.name_scope("Loss_Function"):
            delta = self.label - self.output
            delta = tf.layers.Flatten()(delta)
            self.loss = tf.reduce_mean(tf.norm(delta, ord=2, axis=1, keepdims=False))
        # Optimizer
        with tf.name_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                self.config.trainer.l_rate,
                beta1=self.config.trainer.optimizer_adam_beta1,
                beta2=self.config.trainer.optimizer_adam_beta2,
            )
            # Collect All Variables
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            self.update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope="Denoising"
            )
            with tf.control_dependencies(self.update_ops):
                self.optimizer.minimize(
                    self.loss,
                    var_list=all_variables,
                    global_step=self.global_step_tensor,
                )

        # Summary
        with tf.name_scope("Summary"):
            with tf.name_scope("Loss"):
                tf.summary.scalar("Loss", self.loss, ["loss"])
            with tf.name_scope("Image"):
                tf.summary.image("Input_Image", self.label, 3, ["image"])
                tf.summary.image("Output_Image", self.output, 3, ["image"])

        self.summary_op_im = tf.summary.merge_all("image")
        self.summary_op_loss = tf.summary.merge_all("loss")
        self.summary_all = tf.summary.merge([self.summary_op_im, self.summary_op_loss])

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.log.max_to_keep)
