import os
import numpy as np
import tensorflow as tf

class VanillaGAN(object):
    """ Vanilla Generative Adversarial Networks

    Attribute:
        z_dim: Integer, dimension of noise vector.
        x_dim: Integer, dimension of generated vector.
        g_hidden: An integer or tuple/list of n-integers, specifying the number of hidden neurons in generator.
        d_hidden: An Integer or tuple/list of n-integers, specifying the number of hidden neurons in discriminator.
        g_activation: Function, activation function for last layer of generator. If None, linear activation will be applied.
        sample_noise: Function, sampling noise vectors, pass size(int) and return float matrix.
        sample_dist: Function, sampling vectors from object distribution, pass size(int) and return float matrix.
        plc_training: tf.placeholder, batch normalization mode, either training mode or inference mode.
        plc_z: tf.placeholder, noise vector.
        plc_x: tf.placeholder, vector sampled from object distribution.
        g: Tensor, generator.
        d1: Tensor, discriminator for object distribution.
        d2: Tensor, discriminator for generated distribution.
        g_loss: Tensor, object function for optimizing generator.
        d_loss: Tensor, object function for optimizing discriminator.
        summary: Tensor, merged scalar summary.
    """
    def __init__(self,
                 z_dim=1,
                 x_dim=1,
                 g_hidden=1,
                 d_hidden=1,
                 sample_noise=lambda x: np.random.uniform(size=[x, 1]),
                 sample_dist=lambda x: np.random.normal(size=[x, 1]),
                 g_activation=None):
        """ Vanilla GANs Initializer

        Args:
            z_dim: Integer, dimension of noise vector.
            x_dim: Integer, dimension of object distribution.
            g_hidden: An integer or tuple/list of n-integers, specifying the number of hidden neurons in generator.
            d_hidden: An Integer or tuple/list of n-integers, specifying the number of hidden neurons in discriminator.
            sample_noise: Function, sampling noise vectors, pass size(int) and return float matrix.
            sample_dist: Function, sampling vectors from object distribution, pass size(int) and return float matrix.
            g_activation: Function, activation function for last layer of generator. If None, linear activation will be applied.
        """
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.sample_noise = sample_noise
        self.sample_dist = sample_dist
        self.g_activation = g_activation

        self.plc_training = tf.placeholder(tf.bool)
        self.plc_z = tf.placeholder(tf.float32, [None, z_dim])
        self.plc_x = tf.placeholder(tf.float32, [None, x_dim])

        self.g = self._get_fully_connected_network(self.plc_z, x_dim, g_hidden, g_activation, 'generator')
        self.d1 = self._get_fully_connected_network(self.plc_x, 1, d_hidden, tf.nn.sigmoid, 'discriminator')
        self.d2 = self._get_fully_connected_network(self.g, 1, d_hidden, tf.nn.sigmoid, 'discriminator', reuse=True)

        self.g_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        self.d_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

        self.g_loss = tf.reduce_mean(tf.log(self.d2))
        self.d_loss = tf.reduce_mean(tf.log(self.d1) + tf.log(1 - self.d2))

        summary_g = tf.summary.scalar('g_loss', self.g_loss)
        summary_d = tf.summary.scalar('d_loss', self.d_loss)
        self.summary = tf.summary.merge_all()

    def train(self, sess, flags):
        summary_dir = os.path.join(flags.summary_dir, flags.model_name)
        # checkpoint_dir = os.path.join(flags.checkpoint_dir, flags.model_name)

        writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
        # saver = tf.train.Saver()

        optimizer_g = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1)
        optimizer_d = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1)

        opt_g = optimizer_g.minimize(-self.g_loss, var_list=self.g_weights)
        opt_d = optimizer_d.minimize(-self.d_loss, var_list=self.d_weights)

        sess.run(tf.global_variables_initializer())

        iter = 0
        n_epoch = flags.dataset_size // flags.batch_size
        n_completed_epochs = 0
        is_nan = False

        while n_completed_epochs < flags.epochs:
            iter += 1
            if (iter % n_epoch) == 0:
                n_completed_epochs += 1

            x = self.sample_dist(flags.batch_size)
            z = self.sample_noise(flags.batch_size)

            loss_d, _ = sess.run([self.d_loss, opt_d], feed_dict={self.plc_x: x, self.plc_z: z, self.plc_training: True})
            loss_g, _ = sess.run([self.g_loss, opt_g], feed_dict={self.plc_z: z, self.plc_training: True})

            summary = sess.run(self.summary, feed_dict={self.plc_x: x, self.plc_z: z, self.plc_training: False})
            writer.add_summary(summary, iter)

            if np.isnan(loss_d):
                is_nan = True
                print('d_loss nan in iter {}'.format(iter))
                break

            if np.isnan(loss_g):
                is_nan = True
                print('g_loss nan in iter {}'.format(iter))
                break

        return not is_nan

    def generate(self, sess, size):
        z = self.sample_noise(size)
        return sess.run(self.g, feed_dict={self.plc_z: z, self.plc_training: False})

    def _get_fully_connected_network(self,
                                     input_tensor,
                                     out_dim,
                                     n_hidden,
                                     activation,
                                     name='network',
                                     reuse=False):
        """ Create Fully Connected Networks

        Args:
            input_tensor: Tensor, input tensor.
            out_dim: Integer, dimension of output tensor.
            n_hidden: An integer or tuple/list of n-integers, specifying the number of hidden neurons.
            activation: Function, activation function for last layer. If None, linear activation will be applied.
            name: String, name of network.
            reuse: Boolean, reuse of variable scope.
        """
        with tf.variable_scope(name, reuse=reuse):
            if isinstance(n_hidden, list):
                layer = input_tensor

                for h in n_hidden:
                    layer = tf.layers.dense(layer, h)
                    layer = tf.nn.relu(tf.layers.batch_normalization(layer, training=self.plc_training))

            else:
                layer = tf.layers.dense(input_tensor, n_hidden)
                layer = tf.nn.relu(tf.layers.batch_normalization(layer, training=self.plc_training))

            layer = tf.layers.dense(layer, out_dim)

            if activation is not None:
                layer = activation(layer)

        return layer
