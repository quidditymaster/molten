import os
import numpy as np
import pandas as pd
import scipy
from scipy.misc import imread

import tensorflow as tf

def make_coordinate_sampler(x, lower_bound, upper_bound, npts):
    cx =(npts-1)*(x-lower_bound)/(upper_bound-lower_bound)
    nearest_xi_float = tf.floor(cx+0.5)
    nearest_xi_int = tf.to_int64(nearest_xi_float)
    delta = cx-nearest_xi_float
    
    return {
        "continuous_index":cx,
        "nearest_index":nearest_xi_int,
        "delta":delta,
    }


def make_silhouette(
    x, y,
    bounds,
    npts,
    silhouette_shape,
    delta_footprint,
    radial_basis_function,
):
    nx, ny = silhouette_shape
    min_x, max_x, min_y, max_y = bounds
    x_sampler = make_coordinate_sampler(x, min_x, max_x, nx)
    y_sampler = make_coordinate_sampler(y, min_y, max_y, ny)
    
    point_indexes = []
    x_indexes = []
    y_indexes = []
    footprint_indexes = []
    rbf_weights = []
    
    n_foot = len(delta_footprint)
    for footprint_idx in range(n_foot):
        dx, dy = delta_footprint[footprint_idx]
        x_idx = tf.clip_by_value(x_sampler["nearest_index"] - dx, 0, nx-1)
        y_idx = tf.clip_by_value(y_sampler["nearest_index"] - dy, 0, ny-1)
        r_delta = tf.sqrt((x_sampler["delta"] + dx)**2 + (y_sampler["delta"] + dy)**2)
        weight = radial_basis_function(r_delta)
        
        rbf_weights.append(weight)
        point_indexes.append(tf.range(0, npts, dtype=tf.int64))
        x_indexes.append(x_idx)
        y_indexes.append(y_idx)
        footprint_indexes.append(tf.constant(np.repeat(footprint_idx, npts)))
    
    #join the tensors together
    rbf_weights = tf.concat(concat_dim=0, values=rbf_weights)
    point_indexes = tf.concat(concat_dim=0, values=point_indexes)
    x_indexes = tf.concat(concat_dim=0, values=x_indexes)
    y_indexes = tf.concat(concat_dim=0, values=y_indexes)
    footprint_indexes = tf.concat(concat_dim=0, values=footprint_indexes)
    
    packed_indices = tf.stack([footprint_indexes, point_indexes, x_indexes, y_indexes], axis=1)
    sparse_shape = tf.constant(np.array([n_foot, npts, nx, ny]))
    sparse_opacity = tf.SparseTensor(packed_indices, values=rbf_weights, shape=sparse_shape)
    
    #sum over the footprint and point indexes to get a 2D tensor
    #dense_opacity = tf.sparse_reduce_sum(tf.sparse_reduce_sum_sparse(sparse_opacity, axis=0), axis=0)
    dense_opacity = tf.reduce_sum(tf.sparse_reduce_sum(sparse_opacity, axis=1), axis=0)
    
    silhouette = 1.0 - tf.exp(-0.5*tf.sqrt(dense_opacity))
    return {
        "dense_opacity":dense_opacity,
        "silhouette":silhouette,
    }


def make_delta_footprint(max_radius):
    deltas = []
    n_delt = int(max_radius+1)
    for dx in range(-n_delt, n_delt+1):
        for dy in range(-n_delt, n_delt+1):
            if np.sqrt(dx**2 + dy**2) < max_radius:
                deltas.append((dx, dy))
    return deltas


def make_squared_residual_sum(x1, x2):
    return tf.reduce_sum((x1-x2)**2)


def make_optimizer(metric, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(metric, global_step=global_step)
    return train_op


def load_image(fname, invert=True):
    image = imread(fname).astype(np.float32)
    im_max = np.max(image)
    image /= np.max(image)
    if invert:
        image = 1-image
    return image



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--invert-images", default=False, type=bool)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--npts", default=100000, type=int)
    parser.add_argument("--layout", default="orthogonal")
    parser.add_argument("--rbf-max-radius", default=2.0, type=int)
    parser.add_argument("--rbf-sigma", type=float, default=1.0)
    parser.add_argument("--max-image-size", default=200)
    parser.add_argument("--n-iter", default=100, type=int)
    parser.add_argument("--hot-start")
    
    args = parser.parse_args()
    
    if not args.hot_start is None:
        init_cloud_positions = pd.read_csv(args.hot_start).values
    else:
        init_cloud_positions = np.random.uniform(
            low=-1,
            high=1,
            size=(args.npts, 3)
        )
    #cast to float32
    init_cloud_positions = init_cloud_positions.astype(np.float32)
    
    cloud = tf.Variable(init_cloud_positions, name="cloud")
    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
    
    rbf_footprint = make_delta_footprint(args.rbf_max_radius)
    
    silhouettes = []
    images = []
    for im_fname in args.images:
        fpath = os.path.join(args.image_dir, im_fname)
        image = load_image(fpath, invert=args.invert_images)
        images.append(image)
        if np.max(image.shape) > args.max_image_size:
            raise ValueError("An image is too large with shape {}\n If you really want to use an image this large increase the --max-image-size option".format(image.shape))
    
    if args.layout == "orthogonal":
        if len(images) != 3:
            raise ValueError("orthogonal layout requires exactly 3 images")
        projection_indexes = [(0, 1), (0, 2), (1, 2)]
        for image_idx in range(len(images)):
            image = images[image_idx]
            c1, c2 = projection_indexes[image_idx]
            csil = make_silhouette(
                cloud[:, c1],
                cloud[:, c2],
                bounds=(-1, 1, -1, 1),
                delta_footprint=rbf_footprint,
                npts=args.npts,
                silhouette_shape = image.shape,
                radial_basis_function = lambda r: tf.exp(-(r/args.rbf_sigma)**2)
            )
            silhouettes.append(csil)
    else:
        raise ValueError("option layout={} not understood".format(args.layout))
    
    opt_metric = None
    
    #TODO: put regularization terms here
    
    for im_idx in range(len(images)):
        image = images[im_idx]
        silhouette = silhouettes[im_idx]["silhouette"]
        sq_resid_sum = make_squared_residual_sum(
            image,
            silhouette,
        )
        if opt_metric is None:
            opt_metric = sq_resid_sum
        else:
            opt_metric = opt_metric + sq_resid_sum
    
    train_op = make_optimizer(opt_metric, learning_rate)
    initialize = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(initialize)
        for i in range(args.n_iter):
            print("optimization step {}".format(i))
            sess.run(train_op, feed_dict={learning_rate:args.learning_rate})
        coords = sess.run(cloud)

    #write out the final point cloud
    coord_df = pd.DataFrame({"x":coords[:, 0], "y":coords[:, 1], "z":coords[:, 2]})
    coord_df.to_csv(args.output)
        
