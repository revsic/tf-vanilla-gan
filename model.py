import json
import numpy as np
import tensorflow as tf

class VanillaGAN(object):
    ''' Vanilla Generative Adversarial Networks

    Attribute:
        z_dim: Integer, dimension of noise vector.
        x_dim: Integer, dimension of generated vector.
        g_hidden: An integer or tuple/list of n-integers, specifying the number of hidden neurons in generator.
        d_hidden: An Integer or tuple/list of n-integers, specifying the number of hidden neurons in discriminator.
        learning_rate: Float, learning rate for optimizing model.
        beta1: Float, beta1 value for optimizing model by Adam Optimizer.
        plc_z: tf.placeholder, noise vector.
        plc_x: tf.placeholder, vector sampled from object distribution.
        g: Tensor, generator.
        d_real: Tensor, discriminator for object distribution.
        d_gen: Tensor, discriminator for generated distribution.
        g_loss: Tensor, object function for optimizing generator.
        d_loss: Tensor, object function for optimizing discriminator.
        g_opt: Tensor, optimizing object for generator.
        d_opt: Tensor, optimizing object for discriminator.
        summary: Tensor, merged scalar summary.
    '''
    def __init__(self, z_dim, x_dim, g_hidden, d_hidden, learning_rate, beta1):
        ''' Initializer
        Args:
            z_dim: Integer, dimension of noise vector.
            x_dim: Integer, dimension of generated vector.
            g_hidden: An integer or tuple/list of n-integers, specifying the number of hidden neurons in generator.
            d_hidden: An Integer or tuple/list of n-integers, specifying the number of hidden neurons in discriminator.
            learning_rate: Float, learning rate for optimizing model.
            beta1: Float, beta1 value for optimizing model by Adam Optimizer.
        '''
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.g_hidden = g_hidden
        self.d_hidden = d_hidden

        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.plc_x = tf.placeholder(tf.float32, [None, x_dim])
        self.plc_z = tf.placeholder(tf.float32, [None, z_dim])

        self.g, self.d_real, self.d_gen = self._get_model()
        self.g_loss, self.d_loss = self._get_loss()

        self.g_opt, self.d_opt = self._get_optimizer()
        self.summary = self._get_summary()
        self.ckpt = tf.train.Saver()

    def train(self, sess, z, x):
        sess.run([self.d_opt, self.g_opt], feed_dict={self.plc_z: z, self.plc_x: x})

    def inference(self, sess, obj, z, x=None):
        feed_dict = {self.plc_z: z}
        if x is not None:
            feed_dict[self.plc_x] = x

        return sess.run(obj, feed_dict=feed_dict)

    def dump(self, sess, path):
        self.ckpt.save(sess, path + '.ckpt')

        with open(path + '.json', 'w') as f:
            dump = json.dumps(
                {
                    'z_dim': self.z_dim,
                    'x_dim': self.x_dim,
                    'g_hidden': self.g_hidden,
                    'd_hidden': self.d_hidden,
                    'learning_rate': self.learning_rate,
                    'beta1': self.beta1
                }
            )

            f.write(dump)

    @classmethod
    def load(cls, sess, path):
        with open(path + '.json') as f:
            param = json.loads(f.read())

        model = cls(
            param['z_dim'],
            param['x_dim'],
            param['g_hidden'],
            param['d_hidden'],
            param['learning_rate'],
            param['beta1']
        )
        model.ckpt.restore(sess, path + '.ckpt')

        return model

    def _get_model(self):
        g = self._get_generator(name='gen')
        d_real = self._get_discriminator(self.plc_x, name='disc')
        d_gen = self._get_discriminator(g, name='disc', reuse=True)

        return g, d_real, d_gen

    def _get_loss(self):
        g_loss = tf.reduce_mean(tf.log(self.d_gen))
        d_loss = tf.reduce_mean(tf.log(self.d_real) + tf.log(1 - self.d_gen))

        return g_loss, d_loss

    def _get_optimizer(self):
        g_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')
        d_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc')

        g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(-self.g_loss, var_list=g_weights)
        d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(-self.d_loss, var_list=d_weights)

        return g_opt, d_opt

    def _get_summary(self):
        summary = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('d_loss', self.d_loss)
        ])

        return summary

    def _get_generator(self, name='gen', reuse=False):
        network = self._get_fully_connected_network(self.plc_z,
                                                    self.x_dim,
                                                    self.g_hidden,
                                                    activation=tf.nn.sigmoid,
                                                    name=name,
                                                    reuse=reuse)
        return network

    def _get_discriminator(self, input_tensor, name='disc', reuse=False):
        network = self._get_fully_connected_network(input_tensor,
                                                    1,
                                                    self.d_hidden,
                                                    activation=tf.nn.sigmoid,
                                                    name=name,
                                                    reuse=reuse)
        return network

    def _get_fully_connected_network(self,
                                     input_tensor,
                                     out_dim,
                                     n_hidden,
                                     activation=None,
                                     name='fc',
                                     reuse=False):
        ''' Get Fully Connected Network

        Args:
            input_tensor: Tensor, input tensor of fully connected neural network.
            out_dim: Integer, dimension of output vector.
            n_hidden: An integer of tuple/list of n-integers, specifying the number of hidden neurons.
            activation: Activation function, applied to last layer.
            name: String, name of the variable scope.
            reuse: Bool, reuse of variable scope.

        Returns:
            layer: Tensor, last layer of fully connected network.
        '''
        with tf.variable_scope(name, reuse=reuse):
            if isinstance(n_hidden, list):
                layer = input_tensor

                for h in n_hidden:
                    layer = tf.layers.dense(layer, h, activation=tf.nn.relu)

            else:
                layer = tf.layers.dense(input_tensor, n_hidden, activation=tf.nn.relu)

            layer = tf.layers.dense(layer, out_dim)
            if activation is not None:
                layer = activation(layer)

            return layer
