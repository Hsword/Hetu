import numpy as np


def softmax_func(y):
    """Computes softmax activations.
      This function performs the equivalent of
          softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits))
    another form: np.exp(y)/np.sum(np.exp(y),axis=1,keepdims=True)
    """

    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


def confusion_matrix_at_thresholds(labels, predictions, thresholds, includes=None):
    """Computes true_positives, false_negatives, true_negatives, false_positives.
      Args:
        labels: A np.array whose shape matches `predictions`. Will be cast to
          `bool`.
        predictions: A floating point np.array of arbitrary shape and whose values
          are in the range `[0, 1]`.
        thresholds: A python list or tuple of float thresholds in `[0, 1]`.
        includes: Tuple of keys to return, from 'tp', 'fn', 'tn', fp'. If `None`,
            default to all four.
      Returns:
        values: Dict of variables of shape `[len(thresholds)]`. Keys are from
            `includes`.
      """
    all_includes = ('tp', 'fn', 'tn', 'fp')
    if includes is None:
        includes = all_includes
    else:
        for include in includes:
            if include not in all_includes:
                raise ValueError('Invaild key: %s.' % include)
    # Reshape predictions and labels.
    # This function is often used in dichotomies.
    # In multi-classification problems, we often stretch the dimensions directly into dichotomies.
    predictions_2d = np.reshape(predictions, [-1, 1])
    labels_2d = np.reshape(labels.astype(dtype=np.bool), [1, -1])
    num_predictions = predictions_2d.shape[0]
    num_thresholds = len(thresholds)
    # thresh_tiled's shape:[num_thresholds,num_predictions]
    thresh_tiled = np.tile(
        np.expand_dims(np.array(thresholds), axis=1), [1, num_predictions])
    pred_is_pos = np.greater(
        np.tile(np.transpose(predictions_2d), [num_thresholds, 1]),
        thresh_tiled)
    if ('fn' in includes) or ('tn' in includes):
        pred_is_neg = np.logical_not(pred_is_pos)
    # Tile labels by number of thresholds
    # label_is_pos's shape:[num_thresholds,num_predictions]
    label_is_pos = np.tile(labels_2d, [num_thresholds, 1])
    if ('fp' in includes) or ('tn' in includes):
        label_is_neg = np.logical_not(label_is_pos)

    values = {}
    if 'tp' in includes:
        is_true_positive = np.logical_and(
            label_is_pos, pred_is_pos).astype(np.float32)
        values['tp'] = np.sum(is_true_positive, axis=1)
    if 'fn' in includes:
        is_false_negative = np.logical_and(
            label_is_pos, pred_is_neg).astype(np.float32)
        values['fn'] = np.sum(is_false_negative, axis=1)
    if 'tn' in includes:
        is_true_negative = np.logical_and(
            label_is_neg, pred_is_neg).astype(np.float32)
        values['tn'] = np.sum(is_true_negative, axis=1)
    if 'fp' in includes:
        is_false_positive = np.logical_and(
            label_is_neg, pred_is_pos).astype(np.float32)
        values['fp'] = np.sum(is_false_positive, axis=1)
    return values


def roc_pr_curve(values, curve='ROC'):
    """Computes the roc-auc or pr-auc based on confusion counts.
    Args:
        values: A dict from the func:confusion_matrix_at_thresholds and must have
            four keys:tp,fp,fn,tn
        curve: Specifies the name of the curve to be computed, 'ROC' [default] or
        'PR' for the Precision-Recall-curve.
    Returns:
        x_axis: A python list of the curve's x-axis. In ROC it's fpr;In PR it's Recall.
        y_axis:A python list of the curve's y-axis. In ROC it's tpr;In PR it's Precision.
            fpr=fp/(fp+tn)
            tpr=tp/(tp+fn)
            Recall=tpr
            Precision=tp/(tp+fp)
    """
    if 'tp' not in values.keys():
        raise ValueError('values must have the key tp')
    if 'fp' not in values.keys():
        raise ValueError('values must have the key fp')
    if 'fn' not in values.keys():
        raise ValueError('values must have the key fn')
    if 'tn' not in values.keys():
        raise ValueError('values must have the key tn')
    tp = values['tp']
    fp = values['fp']
    fn = values['fn']
    tn = values['tn']
    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6
    rec = np.divide(tp + epsilon, tp + fn + epsilon)
    if curve == 'ROC':
        fp_rate = np.divide(fp + epsilon, fp + tn + epsilon)
        x_axis = fp_rate
        y_axis = rec
    else:  # curve == 'PR'.
        prec = np.divide(tp + epsilon, tp + fp + epsilon)
        x_axis = rec
        y_axis = prec
    return x_axis, y_axis


def auc(labels, predictions, num_thresholds=200,
        curve='ROC'):
    """Computes the approximate AUC via a Riemann sum.
      We get four variables `true_positives`,`true_negatives`, `false_positives`
      and `false_negatives` that are used to compute the AUC first.
      And then compute auc_curve using the function roc_pr_curve.
      The `num_thresholds` variable controls the degree of discretization with
      larger numbers of thresholds more closely approximating the true AUC.
      For best results, `predictions` should be distributed approximately uniformly
      in the range [0, 1] and not peaked around 0 or 1.
      Args:
        labels: A np.array whose shape matches `predictions`. Will be cast to
          `bool`.
        predictions: A floating point np.array of arbitrary shape and whose values
          are in the range `[0, 1]`.
        num_thresholds: The number of thresholds to use when discretizing the roc
          curve.
        curve: Specifies the name of the curve to be computed, 'ROC' [default] or
        'PR' for the Precision-Recall-curve.
      Returns:
        auc: A scalar representing the current area-under-curve.
      """
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                  for i in range(num_thresholds - 2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
    values = confusion_matrix_at_thresholds(labels, predictions, thresholds)
    x_axis, y_axis = roc_pr_curve(values, curve=curve)
    auc_value = np.sum(np.multiply(
        x_axis[:num_thresholds - 1] - x_axis[1:],
        (y_axis[:num_thresholds - 1] + y_axis[1:]) / 2.))
    return auc_value


def accuracy(labels, predictions):
    """Calculates the degree of `predictions` matches `labels`.
    Args:
        labels: A np.array whose shape matches `predictions`.
        predictions: A floating point np.array of arbitrary shape and it's the
        predicted value.
    returns:
        accuracy: A  accuracy, the value of `total` divided by `count`.
    """
    acc_val = np.equal(
        np.argmax(labels, 1),
        np.argmax(predictions, 1)).astype(np.float32)
    accuracy = np.mean(acc_val)
    return accuracy


def confusion_matrix_one_hot(labels, predictions):
    """Computes true_positives, false_negatives, true_negatives, false_positives.

      Args:
        labels: A np.array whose shape matches `predictions` and must be one_hot.
         Will be cast to `bool`.
        predictions: A floating point np.array of arbitrary shape.
      Returns:
        values: Dict of variables of shape `[predictions.shape[1]]`.
      example:
        labels:[[1,0,0]
                [0,1,0]
                [0,0,1]]
        predictions:
               [[9.1,5.0,7.8]   true
                [0.3,0.7,1.4]   false
                [4.3,1.3,5.3]]  true
        returns:
        values{'tp':[1,0,1]
               'tn':[2,2,1]
               'fp':[0,0,1]
               'fn':[0,1,0]
               }
    """
    # transpose prediction to one hot.for the example above,it will be:
    # [[1,0,0]
    # [0,0,1]
    # [0,0,1]]
    prediction_one_hot = np.eye(predictions.shape[1])[
        np.argmax(predictions, axis=1)]
    values = {}
    is_true_positive = np.logical_and(
        np.equal(labels, True), np.equal(prediction_one_hot, True))
    is_false_positive = np.logical_and(
        np.equal(labels, False), np.equal(prediction_one_hot, True))
    is_true_negatives = np.logical_and(
        np.equal(labels, False), np.equal(prediction_one_hot, False))
    is_false_negatives = np.logical_and(
        np.equal(labels, True), np.equal(prediction_one_hot, False))
    values['tp'] = np.sum(
        is_true_positive.astype(dtype=np.float32), axis=0)
    values['fp'] = np.sum(
        is_false_positive.astype(dtype=np.float32), axis=0)
    values['tn'] = np.sum(
        is_true_negatives.astype(dtype=np.float32), axis=0)
    values['fn'] = np.sum(
        is_false_negatives.astype(dtype=np.float32), axis=0)
    return values


def precision_score_one_hot(labels, predictions, average=None):
    """compute precision score, precision=tp/(tp+fp)
    the labels must be one_hot.
    the predictions is prediction results.
    Args:
        labels: A np.array whose shape matches `predictions` and must be one_hot.
    Will be cast to `bool`.
        predictions: A floating point np.array of arbitrary shape.
        average : string, [None(default), 'micro', 'macro',]
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.

    Returns:
        values:  A score  .

    References
    -----------------------
     [1]   https://blog.csdn.net/sinat_28576553/article/details/80258619
    """
    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6
    values = confusion_matrix_one_hot(labels, predictions)
    if average is None:
        tp = values['tp']
        fp = values['fp']
        p = np.divide(tp+epsilon, tp + fp+epsilon)
        return p
    elif average == 'micro':
        tp = np.sum(values['tp'])
        fp = np.sum(values['fp'])
        return np.divide(tp+epsilon, tp + fp+epsilon)

    elif average == 'macro':
        tp = values['tp']
        fp = values['fp']
        p = np.divide(tp+epsilon, tp + fp+epsilon)
        return np.average(p)
    else:
        raise ValueError('Invaild average: %s.' % average)


def recall_score_one_hot(labels, predictions, average=None):
    """compute recall score, precision=tp/(tp+fn)
    the labels must be one_hot.
    the predictions is prediction results.
    Args:
        labels: A np.array whose shape matches `predictions` and must be one_hot.
    Will be cast to `bool`.
        predictions: A floating point np.array of arbitrary shape.
            average : string, [None(default), 'micro', 'macro',]
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
    Returns:
        values:  A score  .

    References
    -----------------------
     [1]   https://blog.csdn.net/sinat_28576553/article/details/80258619
    """
    # Add epsilons to avoid dividing by 0.
    epsilon = 1.0e-6
    values = confusion_matrix_one_hot(labels, predictions)
    if average is None:
        tp = values['tp']
        fn = values['fn']
        p = np.divide(tp+epsilon, tp + fn+epsilon)
        return p
    elif average == 'micro':
        tp = np.sum(values['tp'])
        fn = np.sum(values['fn'])
        return np.divide(tp+epsilon, tp + fn+epsilon)

    elif average == 'macro':
        tp = values['tp']
        fn = values['fn']
        p = np.divide(tp+epsilon, tp + fn+epsilon)
        return np.average(p)
    else:
        raise ValueError('Invaild average: %s.' % average)


def f_score_one_hot(labels, predictions, beta=1.0, average=None):
    """compute f score, =(1+beta*beta)precision*recall/(beta*beta*precision+recall)
     the labels must be one_hot.
     the predictions is prediction results.
     Args:
         labels: A np.array whose shape matches `predictions` and must be one_hot.
     Will be cast to `bool`.
         predictions: A floating point np.array of arbitrary shape.
             average : string, [None(default), 'micro', 'macro',]
             This parameter is required for multiclass/multilabel targets.
             If ``None``, the scores for each class are returned. Otherwise, this
             determines the type of averaging performed on the data:
             ``'micro'``:
                 Calculate metrics globally by counting the total true positives,
                 false negatives and false positives.
             ``'macro'``:
                 Calculate metrics for each label, and find their unweighted
                 mean.  This does not take label imbalance into account.
     Returns:
         values:  A score float.

     References
     -----------------------
      [1]   https://blog.csdn.net/sinat_28576553/article/details/80258619
     """
    if beta < 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    beta2 = beta ** 2
    p = precision_score_one_hot(labels, predictions, average=average)
    r = recall_score_one_hot(labels, predictions, average=average)
    # In the functions:precision and recall,add a epsilon,so p and r will
    # not be zero.
    f = (1+beta2)*p*r/(beta2*p+r)
    if average is None or average == 'micro':
        p = precision_score_one_hot(labels, predictions, average=average)
        r = recall_score_one_hot(labels, predictions, average=average)
        f = (1 + beta2) * p * r / (beta2 * p + r)
        return f
    elif average == 'macro':
        p = precision_score_one_hot(labels, predictions, average=None)
        r = recall_score_one_hot(labels, predictions, average=None)
        f = (1 + beta2) * p * r / (beta2 * p + r)
        return np.average(f)
    else:
        raise ValueError('Invaild average: %s.' % average)
