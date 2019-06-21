from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from time import sleep
from time import time


class DenoisingTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(DenoisingTrainer, self).__init__(sess, model, data, config, logger)
        # Inititalize the train Dataset Iterator
        self.sess.run(self.data.iterator.initializer)
        # Initialize the test Dataset Iterator
        self.sess.run(self.data.test_iterator.initializer)

    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        begin = time()
        # Attach the epoch loop to a variable
        loop = tqdm(range(self.config.data_loader.num_iter_per_epoch))
        # Define the lists for summaries and losses
        losses = []
        summaries = []
        # Get the current epoch counter
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        image = self.data.test_image
        label = self.data.test_label
        for _ in loop:
            loop.set_description("Epoch:{}".format(cur_epoch + 1))
            loop.refresh()  # to show immediately the update
            sleep(0.01)
            ls, sum_a, = self.train_step(image,label, cur_epoch)
            losses.append(ls)
            summaries.append(sum_a)
        self.logger.info("Epoch {} terminated".format(cur_epoch))
        self.summarizer.add_tensorboard(step=cur_epoch, summaries=summaries)
        loss_m = np.mean(losses)

        self.logger.info(
            "Epoch: {} | time = {} s | loss= {:4f} ".format(cur_epoch, time() - begin, loss_m)
        )
        # Save the model state
        self.model.save(self.sess)
        # Testing

    def train_step(self, image,label, cur_epoch):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        image_eval = self.sess.run(image)
        label_eval = self.sess.run(label)
        feed_dict = {self.model.image_input: image_eval,self.model.label:label_eval}
        loss, sum_a = self.sess.run([self.model.loss, self.model.summary_all], feed_dict=feed_dict)

        return loss, sum_a

