"""
contour loss for Mask R-CNN

This code is suitable for matterport/mask_rcnn
add to model.py
"""

def build_mask_contour_graph(mask):
    """Builds the computation graph of the mask CONTOUR head of mask, use sobel.
    Inputs:
        mask: [N, MASK_POOL_SIZE, MASK_POOL_SIZE, 1]

    Returns: 
        Mask_contour [N, MASK_POOL_SIZE, MASK_POOL_SIZE, 1]
    """
    def sobel_x_init(shape, dtype=tf.float32):
        kernel = tf.constant([
            [-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.]
        ], dtype=tf.float32)
        kernel = tf.expand_dims(kernel, -1)
        kernel = tf.expand_dims(kernel, -1)
        return kernel
    def sobel_y_init(shape, dtype=tf.float32):
        kernel = tf.constant([
            [ 1.,  2.,  1.],
            [ 0.,  0.,  0.],
            [-1., -2., -1.]
        ], dtype=tf.float32)
        kernel = tf.expand_dims(kernel, -1)
        kernel = tf.expand_dims(kernel, -1)
        return kernel

    # Gx = KL.TimeDistributed(KL.Conv2D(1, (3, 3), strides=1, padding="same",
    #                                 kernel_initializer=sobel_x_init),
    #                     name="mrcnn_contour_convX", input_shape = shape)(mask)
    Gx = KL.Conv2D(1, (3, 3), strides=1, padding="same", kernel_initializer=sobel_x_init, name="mrcnn_contour_convX")(mask)
    Gx = KL.Lambda(lambda x: K.abs(x), name="mrcnn_contour_convX_abs")(Gx)
    
    Gy = KL.Conv2D(1, (3, 3), strides=1, padding="same", kernel_initializer=sobel_y_init, name="mrcnn_contour_convY")(mask)
    Gy = KL.Lambda(lambda x: K.abs(x), name="mrcnn_contour_convX_abs")(Gy)
    G = KL.Add()([Gx, Gy])
    G = KL.Lambda(lambda x: K.squeeze(x, 3), name="mrcnn_contour_squeeze")(G)
    return G

def contour_loss(mask_true, mask_pred):
    """
    I have no idea to do what here
    """
    contour_true = build_mask_contour_graph(mask_true)
    contour_pred = build_mask_contour_graph(mask_pred)
    return tf.constant(1.0)

def mrcnn_mask_contour_loss_graph(config, target_masks, target_class_ids, pred_masks):
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    mask_true = tf.gather(target_masks, positive_ix) #[N, h, w]
    mask_pred = tf.gather_nd(pred_masks, indices)

    y_true = tf.expand_dims(mask_true, -1) #[N, h, w, 1]
    y_pred = tf.expand_dims(mask_pred, -1)

    loss = K.switch(tf.size(y_true) > 0,
                    contour_loss(mask_true=y_true, mask_pred=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


contour_loss = KL.Lambda(lambda x: mrcnn_mask_contour_loss_graph(config, *x), name="mrcnn_mask_contour_loss")(
                        [target_mask, target_class_ids, mrcnn_mask])

def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
    """contour layer must be no train
    """
    # Print message on the first call (but not on recursive calls)
    if verbose > 0 and keras_model is None:
        log("Selecting layers to train")

    keras_model = keras_model or self.keras_model

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
        else keras_model.layers

    for layer in layers:
        # Is the layer a model?
        if layer.__class__.__name__ == 'Model':
            print("In model: ", layer.name)
            self.set_trainable(
                layer_regex, keras_model=layer, indent=indent + 4)
            continue

        if not layer.weights:
            continue
        # Is it trainable?
        trainable = bool(re.fullmatch(layer_regex, layer.name))
        # Update layer. If layer is a container, update inner layer.
        if layer.__class__.__name__ == 'TimeDistributed':
            layer.layer.trainable = trainable
            if "contour" in layer.layer.name:
                layer.layer.trainable = False
        else:
            layer.trainable = trainable
            if "contour" in layer.name:
                layer.trainable = False
        # Print trainable layer names
        if trainable and verbose > 0:
            log("{}{:20}   ({})".format(" " * indent, layer.name,
                                        layer.__class__.__name__))