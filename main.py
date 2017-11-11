import os
import model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

flags = tf.app.flags
flags.DEFINE_integer('z_dim', 1, 'Integer, dimension of noise vector, default 1')
flags.DEFINE_integer('x_dim', 1, 'Integer, dimension of object distribution, default 1')
flags.DEFINE_integer('g_hidden', 1, 'Integer, the number of hidden neurons in generator, default 1')
flags.DEFINE_integer('d_hidden', 1, 'Integer, the number of hidden neurons in discriminator, default 1')
flags.DEFINE_integer('batch_size', 128, 'Integer, size of batch, default 128')
flags.DEFINE_integer('dataset_size', 128, 'Integer, size of full dataset, default 128')
flags.DEFINE_integer('epochs', 100, 'Integer, the number of epochs, default 100')
flags.DEFINE_float('learning_rate', 0.01, 'Float, learning rate, default 0.01')
flags.DEFINE_float('beta1', 0.9, 'Float, beta1 value for Adam optimizer, default 0.9')
flags.DEFINE_string('model_name', 'vanilla_gan', 'String, name of model, default vanilla_gan')
flags.DEFINE_string('summary_dir', './log', 'String, directory to save scalar summary, default ./log')
flags.DEFINE_string('checkpoint_dir', './ckpt', 'String, directory to save checkpoint, default ./ckpt')
flags.DEFINE_string('generated_dir', './img', 'String, directory to save generated samples, default ./img')
FLAGS = flags.FLAGS

def main(_):
    def sample_noise(size):
        noise = np.random.uniform(-5, 5, size)
        return np.reshape(sorted(noise), [-1, 1])

    def sample_dist(size):
        noise = np.random.uniform(-5, 5, size)
        normal = norm.pdf(sorted(noise), loc=0, scale=1)
        return np.reshape(normal, [-1, 1])

    gan = model.VanillaGAN(
        z_dim=FLAGS.z_dim,
        x_dim=FLAGS.x_dim,
        g_hidden=[10, 10], #FLAGS.g_hidden,
        d_hidden=[10, 10], #FLAGS.d_hidden,
        sample_noise=sample_noise,
        sample_dist=sample_dist,
        g_activation=tf.nn.sigmoid
    )

    if not os.path.exists(FLAGS.summary_dir):
        os.makedirs(FLAGS.summary_dir)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.generated_dir):
        os.makedirs(FLAGS.generated_dir)

    with tf.Session() as sess:
        success = gan.train(sess, FLAGS)

        if success:
            sample = gan.generate(sess, 100)
            space = np.linspace(-5, 5, 100)
            normal = norm.pdf(space, loc=0, scale=1)

            plt.plot(space, normal, label='dist')
            plt.plot(space, sample, label='generated')
            plt.legend()
            
            path = os.path.join(FLAGS.generated_dir, FLAGS.model_name + '.png')
            plt.savefig(path)

if __name__ == '__main__':
    tf.app.run()
