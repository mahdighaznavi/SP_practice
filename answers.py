import tensorflow as tf


def matmul(A, B, name=None):
    """

    Args:
        A: a 2-D Tensor of shape [m, k].
        B: a 2-D Tensor of shape [k, n].
        name: a name for operation (optional).

    Returns:
        a 2-D tensor of shape [m, n] which is the matrix multiple of A and B.

    Raises:
        ValueError: if A and B are not 2-D Tensors.
        InvalidArgumentError: if A and B are not matrix multipliable.

    """
    if name is None:
        name = "matmul"
    asserts = []
    with tf.name_scope(name):
        asserts.append(tf.assert_rank(A, 2))  # check if A is a 2-dim matrix
        asserts.append(tf.assert_rank(B, 2))  # check if B is a 2-dim matrix
        asserts.append(tf.assert_equal(tf.shape(A)[1], tf.shape(B)[0]))  # check if A and B can be matrix multiplied
        with tf.control_dependencies(asserts):  # check asserts before running the nodes
            C = tf.expand_dims(A, 2) * tf.expand_dims(B, 0)
    return tf.reduce_sum(C, 1, name=name)  # return a node with correct name


def ranges(vector, name=None):
    """

    Args:
        vector: a 1-D integer vector with positive numbers that contain ranges.
        name: a name for operation (optional).

    Returns:
        a 1-D integer vector which is the concatenation of ranges of numbers in vector.

    Raises:
        ValueError: if vector is not a 1-D Tensor.
        TypeError: if vector is not integer type.
        InvalidArgumentError: if elements of vector are not positive.

    """
    if name is None:
        name = "ranges"
    asserts = []
    with tf.name_scope(name):
        asserts.append(tf.assert_rank(vector, 1))  # check if vector is a vector
        asserts.append(tf.assert_integer(vector))  # check if vector are integers
        asserts.append(tf.assert_positive(vector))
        with tf.control_dependencies(asserts):  # check asserts before running the nodes
            max_range = tf.reduce_max(vector)
            max_range_array = tf.expand_dims(tf.range(max_range), 0)
            max_range_array = tf.tile(max_range_array, [tf.size(vector), 1])  # creating ranges 2-D array
            mask = max_range_array < tf.expand_dims(vector, 1)  # creating mask array
    return tf.boolean_mask(max_range_array, mask, name=name)


def negative_sampled_softmax(weights, biases, labels, inputs, num_sampled, num_classes, name=None):
    """

    Args:
      weights: a 2-D Tensor of shape [num_classes, dim]. class embeddings.
      biases: a 1-D Tensor of [num_classes]. class biases
      labels: a 1-D Tensor of type int64 and shape [batch_size]. the target classes.
      inputs: A Tensor of shape [batch_size, dim].
      num_sampled: an integer scalar. the number of classes to randomly sample per batch.
      num_classes: an integer scalar. the number of possible classes.
      name: a name for operation (optional).

    Returns:
        a 1-D Tensor of shape [batch_size] of per-sample softmax losses.

    Raises:
        ValueError: if rank of args does not match
        InvalidArgumentError: if shape of args does not match (like num_classes and ...)
        TypeError: if type of labels is not int64

    """
    if name is None:
        name = "negative_sampled_softmax"
    asserts = []
    with tf.name_scope(name):
        # assert about args rank
        asserts.append(tf.assert_rank(weights, 2))
        asserts.append(tf.assert_rank(biases, 1))
        asserts.append(tf.assert_rank(labels, 1))
        asserts.append(tf.assert_rank(inputs, 2))
        asserts.append(tf.assert_scalar(num_classes))
        asserts.append(tf.assert_scalar(num_sampled))

        # assert about dimensions compatibility
        weights_shape = tf.shape(weights)
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        dim = inputs_shape[1]
        asserts.append(tf.assert_equal(weights_shape[0], num_classes))
        asserts.append(tf.assert_equal(weights_shape[1], dim))
        asserts.append(tf.assert_equal(tf.shape(biases)[0], batch_size))
        asserts.append(tf.assert_equal(tf.shape(labels), batch_size))

        # labels type should be int64
        asserts.append(tf.assert_type(labels, tf.int64))

        with tf.control_dependencies(asserts):
            # creating negative samples
            sample_probs = tf.zeros([batch_size, num_classes])
            negative_samples = tf.multinomial(sample_probs, num_sampled)
            whole_samples = tf.concat([tf.expand_dims(labels, 1), negative_samples], axis=1) # all real and negative
                                                                                             # samples together

            # creating logits and probabilty of being positive for each all samples
            whole_weights = tf.gather(weights, whole_samples)
            whole_biases = tf.gather(biases, whole_samples)
            logits = tf.reduce_sum(tf.expand_dims(inputs, 1) * whole_weights, 2) + whole_biases
            probs = tf.sigmoid(logits) # the probability of being positive for all samples

            # now we should determine the probabilty of being negative sample for negative samples
            # and positive for positive samples
            # then determine the probabilities of the truth (negative or positive) for all samples
            negative_probs = 1 - probs[:, 1:]
            positive_probs = probs[:, 0:1]
            final_probs = tf.concat([positive_probs, negative_probs], 1)
            log_probs = -tf.log(final_probs)  # loss all positive and negative samples
    return tf.reduce_sum(log_probs, 1, name=name)  # negative sampled softmax loss for each sample in a batch





















