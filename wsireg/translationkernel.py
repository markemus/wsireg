"""Basic kernel function. Applies a translation measuring kernel to an image.

Don't be fooled by tensorflow, this is a fixed kernel."""
import numpy as np
import tensorflow as tf


# Config
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Graph
graph = tf.Graph()
with graph.as_default():
    fixed = tf.placeholder(dtype=tf.float32, shape=(199,199))
    moving = tf.placeholder(dtype=tf.float32, shape=(199,199))
    kx = tf.placeholder(dtype=tf.float32, shape=(199,199))
    ky = tf.placeholder(dtype=tf.float32, shape=(199,199))

    fx = tf.reduce_sum(tf.multiply(fixed, kx, name="allfdx"))
    fy = tf.reduce_sum(tf.multiply(fixed, ky, name="allfdy"))

    mx = tf.reduce_sum(tf.multiply(moving, kx, name="allfdx"))
    my = tf.reduce_sum(tf.multiply(moving, ky, name="allfdy"))

    dx = mx - fx
    dy = my - fy

# Images- need fixed, moving.
def full_overlap(source, shiftx, shifty):
    # Use smaller source (100,100) so no unmatched pixels
    dest = np.zeros((199, 199), dtype=np.float32)
    fmap = dest.copy()
    fmap[0:100, 0:100] = source
    mmap = dest.copy()
    mmap[shiftx:100+shiftx, shifty:100+shifty] = source
    return fmap, mmap

if __name__ == "__main__":
    # Signal image
    source = np.random.rand(100,100)
    # # Binary image
    # source = np.random.choice([0.0, 1.0], (100, 100))
    # # Single pixel image
    # source = np.zeros((100,100), dtype=np.float32)
    # source[50,50] = .5
    # source[110,110] = 1

    # Create moving image
    shiftx, shifty = 40, 30
    # mmap = np.roll(fmap, [shiftx, shifty], axis=[0,1])
    fmap, mmap = full_overlap(source, shiftx, shifty)

    # Kernel
    dkern = np.arange(199)-99
    xkern = np.repeat(dkern, repeats=199).reshape(199,199)
    ykern = xkern.T

    # Mask borders?
    # for kern in [xkern, ykern]:
    #     kern[:30, :] = 0
    #     kern[-30:, :] = 0
    #     kern[:, :30] = 0
    #     kern[:, -30:] = 0

    # Run
    sess = tf.Session(graph=graph, config=tf_config)
    x, y = sess.run([dx, dy], feed_dict={fixed: fmap,
                                         moving: mmap,
                                         kx: xkern,
                                         ky: ykern})

    correction = 1/fmap.sum()
    # Test
    print("shiftx: ", shiftx)
    print("shifty: ", shifty)
    print("x estimate: ", x * correction)
    print("y estimate: ", y * correction)
