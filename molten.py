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
        y, x,
        opacity,
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
    
    r_delta_tensors = []
    rbf_weights = []
    for footprint_idx in range(n_foot):
        dx, dy = delta_footprint[footprint_idx]
        r_delta = tf.sqrt((x_sampler["delta"] + dx)**2 + (y_sampler["delta"] + dy)**2)
        r_delta_tensors.append(r_delta)
        weight = radial_basis_function(r_delta)
        rbf_weights.append(weight)
    
    rbf_norm = tf.reduce_sum(tf.pack(rbf_weights), axis=0)
    
    pointwise_opacities = []
    for footprint_idx in range(n_foot):
        dx, dy = delta_footprint[footprint_idx]
        x_idx = tf.clip_by_value(x_sampler["nearest_index"] - dx, 0, nx-1)
        y_idx = tf.clip_by_value(y_sampler["nearest_index"] - dy, 0, ny-1)
        pw_opac = opacity*(rbf_weights[footprint_idx]/rbf_norm)
        pointwise_opacities.append(pw_opac)
        point_indexes.append(tf.range(0, npts, dtype=tf.int64))
        x_indexes.append(x_idx)
        y_indexes.append(y_idx)
        footprint_indexes.append(tf.constant(np.repeat(footprint_idx, npts)))
    
    #join the tensors together
    opac_vec = tf.concat(concat_dim=0, values=pointwise_opacities)
    point_indexes = tf.concat(concat_dim=0, values=point_indexes)
    x_indexes = tf.concat(concat_dim=0, values=x_indexes)
    y_indexes = tf.concat(concat_dim=0, values=y_indexes)
    footprint_indexes = tf.concat(concat_dim=0, values=footprint_indexes)
    
    packed_indices = tf.stack([footprint_indexes, point_indexes, x_indexes, y_indexes], axis=1)
    sparse_shape = tf.constant(np.array([n_foot, npts, nx, ny]))
    sparse_opacity = tf.SparseTensor(packed_indices, values=opac_vec, shape=sparse_shape)
    
    #sum over the footprint and point indexes to get a 2D tensor
    #dense_opacity = tf.sparse_reduce_sum(tf.sparse_reduce_sum_sparse(sparse_opacity, axis=0), axis=0)
    dense_opacity = tf.reduce_sum(tf.sparse_reduce_sum(sparse_opacity, axis=1), axis=0)
    
    silhouette = 1.0 - tf.exp(-tf.sqrt(dense_opacity))
    return {
        "dense_opacity":dense_opacity,
        "silhouette":silhouette,
    }


def make_delta_footprint(max_radius):
    deltas = []
    n_delt = int(max_radius+1)
    for dx in range(-n_delt, n_delt+1):
        for dy in range(-n_delt, n_delt+1):
            if np.sqrt(dx**2 + dy**2) <= max_radius:
                deltas.append((dx, dy))
    return deltas


def make_squared_residual_sum(x1, x2, sigma=None):
    if sigma is None:
        return tf.reduce_sum((x1-x2)**2)
    else:
        return tf.reduce_sum(((x1-x2)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)

def make_optimizer(metric, learning_rate, optimizer):
    if optimizer == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == "GradientDescent":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError("optimizer not recognized")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(metric, global_step=global_step)
    return train_op


def load_image(fname, invert_value, flip_x, flip_y):
    image = imread(fname).astype(np.float32)
    im_max = np.max(image)
    image /= np.max(image)
    if invert_value:
        image = 1-image
    if flip_x:
        image = image[:, ::-1]
    if flip_y:
        image = image[::-1]
    return image



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--invert-images", default=False, type=bool)
    parser.add_argument("--flip-x", default=False, type=bool)
    parser.add_argument("--flip-y", default=True, type=bool)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--npts", default=50000, type=int)
    parser.add_argument("--layout", default="orthogonal")
    parser.add_argument("--rbf-max-radius", default=2.0, type=int)
    parser.add_argument("--rbf-sigma", type=float, default=1.0)
    parser.add_argument("--max-image-size", default=200)
    parser.add_argument("--n-iter", default=100, type=int)
    parser.add_argument("--target-pw-opacity", default=5.0, type=float)
    parser.add_argument("--pw-opacity-sigma", default=2.0, type=float)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--hot-start")
    
    args = parser.parse_args()
    
    init_opacity = np.repeat(args.target_pw_opacity, args.npts).astype(np.float32)
    if not args.hot_start is None:
        start_df = pd.read_csv(args.hot_start)
        init_cloud_positions = start_df[["x", "y", "z"]].values
        if "opacity" in start_df.columns:
            init_opacity = start_df["opacity"].values.astype(np.float32)
    else:
        init_cloud_positions = np.random.uniform(
            low=-1,
            high=1,
            size=(args.npts, 3)
        )
    
    #cast to float32
    init_cloud_positions = init_cloud_positions.astype(np.float32)
    
    cloud = tf.Variable(init_cloud_positions, name="cloud")
    opacity = tf.Variable(init_opacity, name="opacity")
    
    #allowing negative opacities would wreak havoc 
    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
    
    rbf_footprint = make_delta_footprint(args.rbf_max_radius)
    
    silhouettes = []
    images = []
    for im_fname in args.images:
        fpath = os.path.join(args.image_dir, im_fname)
        image = load_image(
            fpath,
            invert_value=args.invert_images,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
        )
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
                opacity=opacity,
                bounds=(-1, 1, -1, 1),
                delta_footprint=rbf_footprint,
                npts=args.npts,
                silhouette_shape = image.shape,
                radial_basis_function = lambda r: tf.exp(-(r/args.rbf_sigma)**2)
            )
            silhouettes.append(csil)
    else:
        raise ValueError("option layout={} not understood".format(args.layout))
    
    #regularization terms
    opt_metric = make_squared_residual_sum(opacity, args.target_pw_opacity, sigma=args.pw_opacity_sigma)
    
    for im_idx in range(len(images)):
        image = images[im_idx]
        silhouette = silhouettes[im_idx]["silhouette"]
        sq_resid_sum = make_squared_residual_sum(
            image,
            silhouette,
        )
        opt_metric = opt_metric + sq_resid_sum
    
    train_op = make_optimizer(opt_metric, learning_rate, optimizer=args.optimizer)
    initialize = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(initialize)
        for i in range(args.n_iter):
            print("optimization step {}".format(i))
            sess.run(train_op, feed_dict={learning_rate:args.learning_rate})
        final_coords = sess.run(cloud)
        final_opacity = sess.run(opacity)
    
    #write out the final point cloud
    coord_df = pd.DataFrame({
        "x":final_coords[:, 0],
        "y":final_coords[:, 1],
        "z":final_coords[:, 2],
        "opacity":final_opacity,
    })
    coord_df.to_csv(args.output)
        
