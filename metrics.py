from keras import backend as K

def pw_prec(num_classes=1):
    def b_pw_prec(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2, 3])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    def c_pw_prec(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    if num_classes == 1:
        return b_pw_prec
    else:
        return c_pw_prec


def pw_recall(num_classes=1):
    return pw_sens(num_classes)


def pw_sens(num_classes=1):
    def b_pw_sens(y_true, y_pred):
        """
        true positive rate, probability of detection

        sensitivity = # of true positives / (# of true positives + # of false negatives)

        Reference: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        :param y_true:
        :param y_pred:
        :return:
        """
        # indices = tf.where(K.greater_equal(y_true, 0.5))
        # y_pred = tf.gather_nd(y_pred, indices)

        y_true = K.round(y_true)
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total_pos = K.sum(K.abs(y_true), axis=[1, 2, 3])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    def c_pw_sens(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total_pos = K.sum(K.abs(y_true), axis=[1, 2])
        return K.mean(true_pos / K.clip(total_pos, K.epsilon(), None), axis=-1)

    if num_classes == 1:
        return b_pw_sens
    else:
        return c_pw_sens

def pw_precesion(num_classes=1):
    def b_pw_precesion(y_true, y_pred):
        """
        true positive rate, probability of detection

        precesion = # of true positives / (# of true positives + # of false positives)

        Reference: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        :param y_true:
        :param y_pred:
        :return:
        """
        # indices = tf.where(K.greater_equal(y_true, 0.5))
        # y_pred = tf.gather_nd(y_pred, indices)

        y_true = K.round(y_true)
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2, 3])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    def c_pw_precesion(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2])
        return K.mean(true_pos / K.clip(total_pos, K.epsilon(), None), axis=-1)

    if num_classes == 1:
        return b_pw_precesion
    else:
        return c_pw_precesion


def pw_spec(num_classes=1):
    """
    true negative rate
    the proportion of negatives that are correctly identified as such

    specificity = # of true negatives / (# of true negatives + # of false positives)

    :param y_true:  ground truth
    :param y_pred: prediction
    :return:
    """

    def b_pw_spec(y_true, y_pred):
        true_neg = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2, 3])
        total_neg = K.sum(K.abs(1. - y_true), axis=[1, 2, 3])
        return true_neg / K.clip(total_neg, K.epsilon(), None)

    def c_pw_spec(y_true, y_pred):
        y_true, y_pred = y_true[..., 1:], y_pred[..., 1:]
        true_neg = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2])
        total_neg = K.sum(K.abs(1. - y_true), axis=[1, 2])
        return true_neg / K.clip(total_neg, K.epsilon(), None)
    if num_classes == 1:
        return b_pw_spec
    else:
        return pw_spec


def dice(num_classes=1):
    def b_dice(y_true, y_pred):
        """
                DSC = (2 * |X & Y|)/ (|X|+ |Y|)
                    = 2 * sum(|A*B|)/(sum(|A|)+sum(|B|))
        :param y_true: ground truth
        :param y_pred: prediction
        :return:
        """

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
        dice = 2 * intersection / K.clip(union, K.epsilon(), None)
        return dice

    def c_dice(y_true, y_pred):

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2])
        dice = 2 * intersection / K.clip(union, K.epsilon(), None)
        return K.mean(dice, axis=-1)

    if num_classes == 1:
        return b_dice
    else:
        return c_dice


def class_jaccard_index(idx):
    def jaccard_index(y_true, y_pred):
        y_true, y_pred = y_true[..., idx], y_pred[..., idx]
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        # Adding all three axis to average across images before dividing
        # See https://forum.isic-archive.com/t/task-2-evaluation-and-superpixel-generation/417/2
        intersection = K.sum(K.abs(y_true * y_pred), axis=[0, 1, 2])
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[0, 1, 2])
        jac = intersection / K.clip(sum_ - intersection, K.epsilon(), None)
        return jac
    return jaccard_index


def iou(num_classes):
    """
    Jaccard index for semantic segmentation, also known as the intersection-over-union.

        This loss is useful when you have unbalanced numbers of pixels within an image
        because it gives all classes equal weight. However, it is not the defacto
        standard for image segmentation.

        For example, assume you are trying to predict if each pixel is cat, dog, or background.
        You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
        should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?

        The loss has been modified to have a smooth gradient as it converges on zero.
        This has been shifted so it converges on 0 and is smoothed to avoid exploding
        or disappearing gradient.

        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        # References

        Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
        What is a good evaluation measure for semantic segmentation?.
        IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.

        https://en.wikipedia.org/wiki/Jaccard_index

        """

    def b_iou(y_true, y_pred):
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
        iou = intersection / K.clip(union - intersection, K.epsilon(), None)
        return iou

    def c_iou(y_true, y_pred):
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        intersection = K.abs(y_true * y_pred)
        union = K.abs(y_true) + K.abs(y_pred)

        intersection = K.sum(intersection, axis=[0, 1, 2])
        union = K.sum(union, axis=[0, 1, 2])

        iou = intersection / K.clip(union - intersection, K.epsilon(), None)
        # iou = K.mean(iou, axis=-1)
        return iou

    if num_classes == 1:
        return b_iou
    else:
        return c_iou



