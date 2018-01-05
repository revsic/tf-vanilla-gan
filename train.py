import os
import model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

flags = tf.app.flags
flags.DEFINE_float('learning_rate', 1e-4, 'Float, learning rate, default 1e-4.')
flags.DEFINE_float('beta1', 0.9, 'Float, beta1 value in Adam, default 0.9.')
flags.DEFINE_integer('train_iter', 50000, 'Integer, number of training iteration, default 5000.')
flags.DEFINE_integer('batch_size', 128, 'Integer, size of batch, default 128.')
flags.DEFINE_integer('ckpt_interval', 5000, 'Integer, interval for writing checkpoint, default 5')
flags.DEFINE_string('name', 'default', 'String, name of model, default `default`.')
flags.DEFINE_string('summary_dir', './summary', 'String, dir name for saving tensor summary, default `./summary`.')
flags.DEFINE_string('ckpt_dir', './ckpt', 'String, dir name for saving checkpoint, default `./ckpt_dir`.')
FLAGS = flags.FLAGS

def get_noise(size):
    noise = np.random.uniform(-5, 5, size)
    return np.reshape(sorted(noise), (-1, 1))

def get_norm(size):
    noise = np.random.uniform(-5, 5, size)
    obj = norm.pdf(sorted(noise), loc=0, scale=1)
    return np.reshape(obj, (-1, 1))

def main(_):
    ckpt_path = os.path.join(FLAGS.ckpt_dir, FLAGS.name)

    with tf.Session() as sess:
        vanilla_gan = model.VanillaGAN(1, 1, 10, 10, FLAGS.learning_rate, FLAGS.beta1)
        writer = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, FLAGS.name), sess.graph)

        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.train_iter):
            batch_z = get_noise(FLAGS.batch_size)
            batch_x = get_norm(FLAGS.batch_size)

            vanilla_gan.train(sess, batch_z, batch_x)

            summary = vanilla_gan.inference(sess, vanilla_gan.summary, batch_z, batch_x)
            writer.add_summary(summary, global_step=i)

            if (i + 1) % FLAGS.ckpt_interval == 0:
                vanilla_gan.dump(sess, ckpt_path)

                space = np.linspace(-5, 5, 100).reshape(-1, 1)
                sample = vanilla_gan.inference(sess, vanilla_gan.g, space)
                disc = vanilla_gan.inference(sess, vanilla_gan.d_real, space, sample)

                plt.plot(space, norm.pdf(space, loc=0, scale=1), label='dist')
                plt.plot(space, sample, label='generated')
                plt.plot(space, disc, label='probability')
                plt.legend()

                path = os.path.join(FLAGS.ckpt_dir, '{}_{}.png'.format(FLAGS.name, i))
                plt.savefig(path)
                plt.clf()

if __name__ == '__main__':
    tf.app.run()