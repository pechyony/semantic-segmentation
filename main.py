import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import numpy as np
import sys
import project_tests as tests

# hyperparameters
learning_rate = 0.001 
reg_rate = 0.0001
keep_probability = 0.5
batch_size = 8
epochs = 200
no_improvement = 10

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer3, layer4, layer7


def layers(pool3_out, pool4_out, layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # convert from original depth=4096 to the depth=number of classes
    conv_1x1 = tf.layers.conv2d(layer7_out, num_classes, kernel_size=1,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
    
    # first upsampling layer
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, kernel_size=4, strides=2, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
    
    # scaled skip connection to the second upsampling layer
    pool4_out_scaled = tf.multiply(pool4_out, 0.01, name='pool4_out_scaled')
    pool4_out_scaled = tf.layers.conv2d(pool4_out_scaled, num_classes, kernel_size=1, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))  
    output = tf.add(output, pool4_out_scaled)

    # second upsampling layer
    output = tf.layers.conv2d_transpose(output, num_classes, kernel_size=4, strides=2, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))

    # scaled skip connection to the third upsampling layer
    pool3_out_scaled = tf.multiply(pool3_out, 0.0001, name='pool3_out_scaled')
    pool3_out_scaled = tf.layers.conv2d(pool3_out_scaled, num_classes, kernel_size=1, 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0)) 
    output = tf.add(output, pool3_out_scaled)

    # third upsampling layer
    output = tf.layers.conv2d_transpose(output, num_classes, kernel_size=16, strides=8, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))

    return output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # define cross-entropy loss
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # collect previously defined regulartization terms
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # define optimizer and objective function 
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss + reg_rate * sum(reg_losses))

    return logits, training_operation, cross_entropy_loss


def train_nn(sess, get_batches_fn, train_op, cross_entropy_loss, logits, input_image,
             correct_label, keep_prob, learning_rate, models_dir, num_classes, image_shape):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param logits: TF Tensor for logits
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param models_dir: Name of direcotry where the models are saved
    :param num_classes: Number of classes used in segmentation
    :param image_shape: Shape of the image
    """

    # initialize saver of models
    saver = tf.train.Saver(max_to_keep=200)

    # the computation of IOU metrics is based on the code from https://steemit.com/machine-learning/@ronny.rest/avoiding-headaches-with-tf-metrics
    ground_truth = tf.placeholder(tf.int32, image_shape)
    prediction = tf.placeholder(tf.int32, image_shape)    
    iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes, name="iou_metric")
    metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="iou_metric")
    metric_vars_initializer = tf.variables_initializer(var_list=metric_vars)

    # initializations
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    best_iou = 0
    best_epoch = -1
    ce_losses = np.zeros(epochs)
    iou_scores = np.zeros(epochs)

    for epoch in range(epochs):

        # training
        for images, labels in get_batches_fn(batch_size):
            sess.run(train_op, feed_dict={keep_prob: keep_probability, input_image: images, correct_label: labels})

        # evaluation - cross-entropy loss over the training set
        total_loss = 0
        n_examples = 0

        for images, labels in get_batches_fn(batch_size):
            loss = sess.run(cross_entropy_loss, feed_dict={keep_prob:1, input_image: images, correct_label: labels})
            total_loss += loss * images.shape[0]
            n_examples += images.shape[0]

        total_loss /= n_examples
        ce_losses[epoch] = total_loss

        # update IOU metric
        iou_scores[epoch] = compute_iou(sess, keep_prob, input_image, logits, correct_label, ground_truth, 
                                        prediction, iou, iou_op, metric_vars_initializer, get_batches_fn, image_shape)

        print("epoch = {} cross-entropy error = {:.4f} IOU = {:.4f}".format(epoch, ce_losses[epoch], iou_scores[epoch]))
        sys.stdout.flush()

        # check if this is the best model obtained so far
        if iou_scores[epoch] > best_iou:
            best_iou = iou_scores[epoch]
            best_epoch = epoch

        # save the current model
        saver.save(sess, '{}/fcn_model_{}'.format(models_dir,epoch))
        
        # check if the training process can be stopped 
        if epoch - best_epoch == no_improvement:
            print("No improvement for the last {} iterations. Stopping...".format(no_improvement))
            break

    print("Best IOU score = {:.4f}".format(best_iou))

    # retrieve the best model
    saver.restore(sess, '{}/fcn_model_{}'.format(models_dir,best_epoch))


def compute_iou(sess, keep_prob, input_image, logits, correct_label, ground_truth, prediction, iou, iou_op, metric_vars_initializer, get_batches_fn, image_shape):
    """
    Compute average IOU over a stream of images
    :param sess: TF Session
    :param keep_prob: TF Placeholder for dropout keep probability
    :param input_image: TF Placeholder for input images
    :param logits: TF Tensor for logits
    :param correct_label: TF Placeholder for label images
    :param ground_truth: TF Placeholder for ground truth segmentation
    :param prediction: TF Placeholder for predicted segmentation
    :param iou: TF operation for computing final value of IOU
    :param iou_op: TF operation for updating IOU
    :param metric_vars_initializer: TF initializer of the computation of IOU
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param image_shape: Shape of the image
    """

    sess.run(metric_vars_initializer)
    for images, labels in get_batches_fn(batch_size):

        # predict class for each pixel of each image in the batch
        predicted_prob = sess.run(tf.nn.softmax(logits), feed_dict={keep_prob:1, input_image: images, correct_label: labels})
        iou_shape = (images.shape[0],image_shape[0],image_shape[1])
        predicted_prob = predicted_prob[:,1].reshape(iou_shape)
        predicted_label = (predicted_prob > 0.5).astype(float)

        # prepare labels for iou computation
        labels = labels[:,:,:,1].reshape(iou_shape).astype(float)

        # compute iou separately for each image
        for i in range(images.shape[0]):
            sess.run(iou_op, feed_dict={ground_truth: labels[i,:,:], prediction: predicted_label[i,:,:]})

    # compute average iou
    iou_score = sess.run(iou)

    return iou_score 


def run():

    image_shape = (160, 576)
    num_classes = 2

    data_dir = './data'     # folder of datasets
    runs_dir = './runs'     # folder of scoring results
    models_dir = './models' # folder of models

    # Check if KITTI dataset exists locally
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Make folder to store models
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Load VGG network
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # Define architecture of fully convolutional network
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        # Define objective function
        label_placeholder = tf.placeholder(tf.int32, (None, None, None, num_classes))
        logits, train_op, cross_entropy_loss = optimize(layer_output, label_placeholder, learning_rate, num_classes)

        # Train neural network 
        train_nn(sess, get_batches_fn, train_op, cross_entropy_loss, logits, input_image,
                 label_placeholder, keep_prob, learning_rate, models_dir, num_classes, image_shape)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
