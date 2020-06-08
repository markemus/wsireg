"""Linear neural net for stain color deconvolution."""
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.constraints import non_neg
from imagetools import nettools as ntls
from imagetools import datamanager as dm
from imagetools import convertimage as ci

# TODO deconv currently doesn't work because it can pick from many arbitrary vectors on
#  the plane where the cone is >= target cone.
#  Needs a cost. Minimize sum of latent values? IE "shortest total distance"
class ColorAutoencoder(ntls.Netbase):
    """input: a set of pixels in optical density space, in long form."""
    def __init__(self, version, save_mode="pretrained"):
        self.name = "ColorDeconv"
        super().__init__(name=self.name, version=version, save_mode=save_mode,
                         ckpt_root=os.path.abspath(__file__ + "/../../data/ckpt/"),
                         tb_log_root=os.path.abspath(__file__ + "/../../data/tb_log/"))

    def build_graph(self, latent_dim=2, device="GPU:1"):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="input")
            self.input_labels = tf.placeholder(tf.int32, shape=[None], name="input_labels")
            # self.latent = tf.layers.dense(inputs=self.input, units=latent_dim, activation=None, name="latent", kernel_constraint=non_neg(), use_bias=False)

            # Latent represents the percentage of each stain present.
            # self.latent_logits = tf.Variable(initial_value=tf.random_normal(shape=[3,2]), name="latent_logits", trainable=True)
            self.latent_logits = tf.Variable(initial_value=tf.random_normal(shape=[3,2]), name="w_latent", trainable=True, constraint=non_neg())
            # self.w_latent = tf.nn.softmax(self.latent_logits, axis=0,  name="w_latent")
            self.w_latent = self.latent_logits
            self.latent = tf.matmul(self.input, self.w_latent, name="latent")

            self.recon = tf.layers.dense(inputs=self.latent, units=3, activation=None, name="recon", use_bias=False, kernel_constraint=non_neg())

            self.recon_loss = tf.losses.mean_squared_error(self.input, self.recon)
            # self.latent_loss = tf.math.reduce_sum(tf.matmul(self.w_latent, self.w_latent, transpose_b=True), name="latent_loss")
            # self.latent_loss = -tf.math.reduce_sum((self.w_latent[:,0]**2) - (self.w_latent[:,1]**2), name="latent_loss")
            # self.latent_loss = tf.losses.mean_squared_error(self.w_latent[:,0], self.w_latent[:,1])
            # self.latent_loss = tf.math.reduce_sum((self.w_latent[:,0] - self.w_latent[:,1])**2)

            # TODO L2 loss should work.
            # Our target vectors should have the smallest squared sum of all possible latent vectors (this will not affect recon loss).
            # self.latent_loss = tf.math.reduce_sum(self.latent**2)
            self.latent_loss = tf.nn.l2_loss(self.latent, name="latent_l2")
            self.loss = self.recon_loss + (.000001 * self.latent_loss)
            # self.loss = self.recon_loss
            self.optimizer = tf.train.AdamOptimizer().minimize(loss=self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss=self.loss)
            # tf.train.AdagradOptimizer.minimize()

            with tf.name_scope('summaries'):
                tf.summary.scalar('train_loss', self.loss)
                tf.summary.scalar('latent_loss', self.latent_loss)
                tf.summary.image('input', tf.reshape(self.input, [-1, 1, 3, 1]))
                tf.summary.image('latent', tf.reshape(self.latent, [-1, 1, 2, 1]))
                # tf.summary.image('latent_1', tf.reshape(self.latent[:,1:], [-1, 1, 1, 1]))
                tf.summary.image('recon', tf.reshape(self.recon, [-1, 1, 3, 1]))
                self.merged_summaries = tf.summary.merge_all()
                self.v_merged_summaries = tf.summary.scalar(name="validation_loss", tensor=self.loss)

    def predict(self, image):
        # k_latent = self.graph.get_tensor_by_name(os.path.split(self.latent.name)[0] + '/kernel:0')
        k_latent = self.graph.get_tensor_by_name("w_latent:0")
        latent, w_latent, recon = self._predict(image, fetches=[self.latent, k_latent, self.recon])

        return latent, w_latent, recon


def image_loader(Netclass, version, img, **kwargs):
    """Loads image pixel data."""
    net = Netclass(version=version, **kwargs)
    net = ntls.split_sets(net, images=img.s_od[0], labels=img.s_od[0,:,0])
    net.img = img.od.reshape(img.shape)

    return net

# TODO create an OOP interface that duplicates the features of this function for testing.
# TODO should threshold low OD values (remove from training set)- see Macenko paper.
def deconv(img, version, epochs, ideal_stains=None, verbose=True, out="../data/out/deconv", new_net=False):
    """Deconvolves a single image and returns the channels + reconstruction (for validation).
    Deconvolution uses a pre-trained neural net that is overfitted to the particular image before
    prediction."""

    # Deconvolve
    if new_net:
        net = image_loader(Netclass=ColorAutoencoder, version=version, img=img, save_mode=None)
        net.build_graph(device="GPU:1")
        net.train(epochs=epochs, continue_training=False, batch_size=1000)
    else:
        net = image_loader(Netclass=ColorAutoencoder, version=version, img=img, save_mode="pretrained")
        net.build_graph(device="GPU:1")
        net.train(epochs=epochs, continue_training=True, batch_size=1000)

    channels, w_latent, recon = net.predict(img.od[0])

    # TODO-DECIDE order stains by target stains? A lot of complexity for perhaps a non-existent problem.
    if ideal_stains is not None:
        # Make sure stains are in a standard order, so that background and foreground images are properly labeled.
        order = order_stains(w_latent, ideal_stains)
        # Quality check- ensures found stains are reasonably close to ideal stains.
        if len(order) != len(set(order)):
            raise ValueError("Identified stains are too divergent from ideal_stains to order!")
        recon = recon[:, order].reshape(img.shape)

    recon = recon.reshape(img.shape)
    channels = channels.reshape(img.shape[:-1] + (2,))

    if verbose:
        signal_0 = channels[:, :, 0]
        signal_1 = channels[:, :, 1]

        # Plot
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].set(title="Input")
        axes[0, 0].imshow(img.od[0].reshape(img.shape))
        axes[0, 1].set(title="Recon")
        axes[0, 1].imshow(recon)
        axes[1, 0].set(title="Latent[0]")
        axes[1, 0].imshow(signal_0, cmap="gray")
        axes[1, 1].set(title="Latent[1]")
        axes[1, 1].imshow(signal_1, cmap="gray")
        fig.show()

        # Save
        norm_recon = (ci.scale_by_max(recon) * 255).astype(int)
        norm_s0 = (ci.scale_by_max(signal_0) * 255).astype(int)
        norm_s1 = (ci.scale_by_max(signal_1) * 255).astype(int)
        cv2.imwrite(os.path.join(out, "%s_recon.jpg" % img.name), norm_recon)
        cv2.imwrite(os.path.join(out, "%s_s0.jpg" % img.name), norm_s0)
        cv2.imwrite(os.path.join(out, "%s_s1.jpg" % img.name), norm_s1)

    return channels, w_latent, recon, net

def deconv_all(paths, pathmode="file", epochs=100, verbose=False, new_net=False):
    """Deconvolves a set of images. Returns the signal and background channels.
    The images must all share the same stain for each channel.

    Epochs: number of training cycles to fit each image."""
    background = []
    signal = []
    stains = []

    for path in paths:
        img = dm.ImageWithSample(path, pathmode=pathmode)
        # channels, recon = deconv(img, epochs=epochs, version="5-non_negative", verbose=verbose)
        channels, w_latent, recon, net = deconv(img, epochs=epochs, version="7-lat_loss", verbose=verbose, new_net=new_net)
        background.append(channels[:, :, 0])
        signal.append(channels[:, :, 1])
        stains.append(w_latent)

    return background, signal, stains

def order_stains(stains, ideal_stains):
    """Ensures consistent order for deconvolved stains."""
    order = [np.argmin(abs((stains.T - i_stain).sum(axis=1))) for i_stain in ideal_stains.T]

    return order

if __name__ == "__main__":
    paths = glob.glob("../data/in/reg?.png")
    bg, s = deconv_all(paths, epochs=100)
    for img in bg:
        plt.imshow(img, vmin=0, vmax=1)
        plt.show()
    print("Deconv finished.")
