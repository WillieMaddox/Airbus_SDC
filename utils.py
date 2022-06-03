import os
import numpy as np
from datetime import datetime
from skimage.io import imread
from skimage.segmentation import find_boundaries
from skimage.morphology import label
from skimage.filters import gaussian
from tqdm import tqdm
import h5py
import cv2

EPS = np.finfo(np.float32).eps


def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M_%S'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


def create_hdf5_dataset(data_dir=None, outfile=None):
    if data_dir in (None, '.'):
        data_dir = os.path.join(os.getcwd(), 'data', 'train_768')
    # if data_dir is None:
    #     data_dir = "/media/Borg_LS/DATA/geos/airbus/input/train_768"
    if outfile is None:
        outfile = data_dir + ".h5"

    img_ids = os.listdir(data_dir)

    with h5py.File(outfile, 'w') as h5:
        for img_id in tqdm(img_ids):
            img = cv2.imread(os.path.join(data_dir, img_id))
            h5.create_dataset(img_id, data=img)


def quick_stats(arr):
    print(arr.shape, arr.dtype, np.min(arr), np.max(arr), np.mean(arr), np.std(arr), np.sum(arr))


def holt_winters_second_order_ewma(x, span, beta):
    # Ref http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
    N = x.size
    alpha = 2.0 / (1 + span)
    s = np.zeros((N,))
    b = np.zeros((N,))
    s[0] = x[0]
    for i in range(1, N):
        s[i] = alpha * x[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    return s


def reversed_recombined_holt_winters(x, span=15, beta=0.3):
    # take EWMA in both directions with a smaller span term
    fwd = holt_winters_second_order_ewma(x, span, beta)
    bwd = holt_winters_second_order_ewma(x[::-1], span, beta)
    c = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
    c = np.mean(c, axis=0)  # average
    return c


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def rle_test():
    for i in range(100):
        data = np.random.randint(0, 2, (100, 100))
        data_rle_enc = rle_encode(data)
        data_rle_dec = rle_decode(data_rle_enc, data.shape)
        np.testing.assert_allclose(data, data_rle_dec)


def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def multi_rle_encode2(img):
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels == k, axis=2)) for k in np.unique(labels[labels > 0])]
    else:
        return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def rle_to_flat_mask(rle_list, shape=(768, 768)):
    """
    Convert a list of run-length encoded masks to a binary mask.
    This assumes that there are no overlapping segmentations.

    :param rle_list:
    :param shape:
    :return:
    """
    flat_mask = np.zeros(shape=shape, dtype=np.uint8)
    for mask in rle_list:
        if isinstance(mask, str):
            flat_mask += rle_decode(mask, shape=shape)
    return flat_mask


def rle_to_label_mask(rle_list, shape=(768, 768)):
    """
    Convert a list of run-length encoded masks to a label mask.
    Create a mask where the values are the labels to each item in the mask list.
    This assumes that there are no overlapping segmentations.

    :param rle_list:
    :param shape:
    :return:
    """
    label_mask = np.zeros(shape=shape, dtype=np.uint8)
    for i, mask in enumerate(rle_list):
        if isinstance(mask, str):
            label_mask += (rle_decode(mask, shape=shape) * (i + 1))
    return label_mask


def rle_to_stacked_masks(rle_list, shape=(768, 768)):
    """
    Convert a list of run-length encoded masks to an array of stacked masks.
    The length of the first dimension is equal to the number of masks in the mask_list.
    This assumes that there are no overlapping segmentations.

    :param rle_list:
    :param shape:
    :return:
    """
    stacked_masks = np.zeros((len(rle_list), *shape), dtype=np.uint8)
    for i, mask in enumerate(rle_list):
        if isinstance(mask, str):
            stacked_masks[i] = rle_decode(mask, shape=shape)
    return stacked_masks


def rle_to_full_mask(rle_list, shape=(768, 768)):
    all_masks = rle_to_flat_mask(rle_list, shape=shape)
    return np.array(all_masks, dtype=bool)


def rle_to_border_mask(rle_list, shape=(768, 768), mode='thick'):
    label_mask = rle_to_label_mask(rle_list, shape=shape)
    bnd = find_boundaries(label_mask, mode=mode)
    return bnd


def rle_to_inner_masks(rle_list, shape=(768, 768), threshold=3):
    """
    The higher the threshold, the larger the area to border ratio will need to be
    for a ship to use its inner mask.

    :param rle_list:
    :param shape:
    :param threshold:
    :return:
    """
    stacked_masks = rle_to_stacked_masks(rle_list)
    inner_bounds = np.zeros((len(rle_list), *shape), dtype=np.uint8)
    for ii, mask in enumerate(stacked_masks):
        bnd_in = find_boundaries(mask, mode='inner').astype(np.uint8)
        n_msk = np.count_nonzero(mask)
        n_bnd_in = np.count_nonzero(bnd_in)
        if n_msk > threshold * n_bnd_in:
            inner_bounds += bnd_in
    return stacked_masks - inner_bounds


def masks_as_image(rle_list, shape=(768, 768)):
    all_masks = rle_to_flat_mask(rle_list, shape=shape)
    return np.expand_dims(all_masks, -1)


def get_mask_weights(rle_list, shape=(768, 768)):
    label_mask = rle_to_label_mask(rle_list, shape=shape)
    bnd = find_boundaries(label_mask, mode='outer') * 1.0
    weights = gaussian(bnd, sigma=2)
    return np.expand_dims(weights, -1)


def read_stacked_masks(img_name, df, shape=(768, 768)):
    rle_list = df.loc[df['ImageId'] == img_name, 'EncodedPixels'].tolist()
    stacked_masks = rle_to_stacked_masks(rle_list, shape=shape)
    return stacked_masks


def read_flat_mask(img_name, df, shape=(768, 768)):
    stacked_masks = read_stacked_masks(img_name, df, shape=shape)
    return np.sum(stacked_masks, axis=0)


def make_image_gen(in_df, image_dir, batch_size=16, print_meta=False):
    all_batches = list(in_df.groupby('ImageId'))
    cnt = 0
    out_rgb = []
    out_mask = []
    mask_cache = {}
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            c_img = imread(os.path.join(image_dir, c_img_id))

            if c_img_id not in mask_cache:
                mask_cache[c_img_id] = masks_as_image(c_masks['EncodedPixels'].values)
            c_mask = mask_cache[c_img_id]

            # c_full = rle_to_full_mask(c_masks['EncodedPixels'].values)
            # c_inner = rle_to_border_mask(c_masks['EncodedPixels'].values, mode='inner')
            # c_outer = rle_to_border_mask(c_masks['EncodedPixels'].values, mode='outer')

            out_rgb += [c_img]
            out_mask += [c_mask]
            if print_meta:
                print(cnt, c_img_id, c_masks['counts'].values, len(mask_cache))
                cnt += 1
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                cnt, out_rgb, out_mask = 0, [], []


def image2label(seg):
    # store the mask and borders in grayscale pixels.
    # ~ 0.5 are borders and ~ 1.0 are the interior segmentation mask.
    # the resulting labels, y, are an (n x n x 2) array.
    msk_bdr = seg > 0.25
    msk = seg > 0.75
    bdr = msk_bdr ^ msk
    return np.concatenate([msk, bdr], axis=-1) * 1.0


def image2label3(seg):
    # store the mask and borders in grayscale pixels.
    # ~ 0.5 are borders and ~ 1.0 are the interior segmentation mask and 0.0 is everything else.
    # the resulting labels, y, are an (n x n x 3) array.
    msk_bdr = seg >= 0.25
    msk = seg >= 0.75
    bdr = msk_bdr ^ msk
    oth = seg < 0.25
    return np.concatenate([msk, bdr, oth], axis=-1) * 1.0


def perf_measure0(y_true, y_pred, axis=-1):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    TP = np.sum(yt & yp, axis=axis)
    TN = np.sum(~yt & ~yp, axis=axis)
    FP = np.sum(~yt & yp, axis=axis)
    FN = np.sum(yt & ~yp, axis=axis)
    return TP, FP, TN, FN


def perf_measure(y_true, y_pred, axis=-1):
    TP = np.sum(y_true * y_pred, axis=axis)
    TN = np.sum((1 - y_true) * (1 - y_pred), axis=axis)
    FP = np.sum((1 - y_true) * y_pred, axis=axis)
    FN = np.sum(y_true * (1 - y_pred), axis=axis)
    return TP, FP, TN, FN


def confusion_matrix_variants(TP, FP, TN, FN):
    TPR = TP / (TP + FN)  # Sensitivity, hit rate, recall, or true positive rate
    TNR = TN / (TN + FP)  # Specificity or true negative rate
    PPV = TP / (TP + FP)  # Precision or positive predictive value
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # Fall out or false positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # Overall accuracy

    #     print(f'{TP:>5}, {TN:>6}, {FP:>5}, {FN:>5}, {TPR:>.5f}, {TNR:>.5f}, {PPV:>.5f}, {NPV:>.5f}, {FPR:>.5f}, {FNR:>.5f}, {FDR:>.5f}, {ACC:>.5f}')
    return TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC


def recall(y_true, y_pred, axis=-1, smooth=1e-3):
    return (np.sum(y_true * y_pred, axis=axis) + smooth) / (np.sum(y_true, axis=axis) + smooth)


def fbeta(y_true, y_pred, beta=2, axis=-1, smooth=1e-3):
    tp, fp, tn, fn = perf_measure(y_true, y_pred, axis=axis)
    return ((beta ** 2 + 1) * tp + smooth) / ((beta ** 2 + 1) * tp + beta ** 2 * fn + fp + smooth)


def iou(y_true, y_pred, axis=-1):
    i = np.sum((y_true * y_pred) > 0.5, axis=axis) + EPS
    u = np.sum((y_true + y_pred) > 0.5, axis=axis) + EPS
    return i / u


IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def f2score(mask_true, mask_prob):

    f2_total = 0
    for th in IOU_THRESHOLDS:
        mask_pred = mask_prob > th
        tp, fp, _, fn = perf_measure0(mask_true, mask_pred)
        f2_total += (5 * tp + EPS) / (5 * tp + 4 * fn + fp + EPS)

    return f2_total / len(IOU_THRESHOLDS)


def bce(y_true, y_pred, **kwargs):
    y_pred = np.clip(y_pred, EPS, 1. - EPS)
    out = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return np.mean(out, axis=-1)


def soft_dice_coef(y_true, y_pred, axis=-1, smooth=1e-3):
    AB = np.sum(y_true * y_pred, axis=axis)
    A = np.sum(y_true, axis=axis)
    B = np.sum(y_pred, axis=axis)
    return (2 * AB + smooth) / (A + B + smooth)


def soft_dice_loss(y_true, y_pred, axis=-1, smooth=1e-3):
    return 1 - soft_dice_coef(y_true, y_pred, axis=axis, smooth=smooth)


# https://www.jeremyjordan.me/semantic-segmentation/
def soft_dice_loss2(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    return 1 - numerator / (denominator + epsilon)


def bce_soft_dice_loss_n0(y_true, y_pred, bce_weight=0.5):
    return bce(y_true, y_pred) * bce_weight + soft_dice_loss(y_true, y_pred) * (1 - bce_weight)


def focal_loss_wrapper(gamma=2., alpha=.25, axis=-1):
    def focal_loss(y_true, y_pred):
        y_pred_c = np.clip(y_pred, EPS, 1. - EPS)
        pt_1 = np.where(np.equal(y_true, 1), y_pred_c, np.ones_like(y_pred))
        pt_0 = np.where(np.equal(y_true, 0), y_pred_c, np.zeros_like(y_pred))
        res1 = alpha * np.power(1. - pt_1, gamma) * np.log(pt_1)
        res0 = (1 - alpha) * np.power(pt_0, gamma) * np.log(1. - pt_0)
        return -np.mean(res1 + res0, axis=axis)
    return focal_loss


def focal_soft_dice_loss_wrapper(gamma=2., alpha=.25, focal_coef=0.5, axis=-1, smooth=1e-3):
    focal_loss = focal_loss_wrapper(gamma=gamma, alpha=alpha, axis=axis)

    def focal_soft_dice_loss(y_true, y_pred):
        return focal_loss(y_true, y_pred) * focal_coef + soft_dice_loss(y_true, y_pred, axis=axis, smooth=smooth) * (1. - focal_coef)

    return focal_soft_dice_loss


def normalized_hamming_distance(hash1, hash2):
    return np.mean((hash1 != hash2) * 1)


def random_crop(img, crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def crop_generator(batch_generator, crop_length):
    """
    Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator
    """
    while True:
        batch_x, batch_y = next(batch_generator)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, batch_x.shape[-1]))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


def get_gaussian_mask(dim, scale=1.0, mu=0.0, sigma=1.0):
    c = dim[0] // 2
    offset = c - 0.5 if dim[0] % 2 == 0 else c
    xyv = np.mgrid[-offset:offset + 1, -offset:offset + 1] / offset
    d = np.linalg.norm(xyv, axis=0)
    g = scale * np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


def get_best_model_name(run_dir):
    best_model = None
    min_loss = 999.9
    run_dir2 = os.path.join('out', run_dir)
    for filename in os.listdir(run_dir2):
        if not filename.endswith('.hdf5'):
            continue
        if '.last.' in filename:
            continue
        filebase = filename.rsplit('.', maxsplit=1)[0]
        loss = float(filebase.split('-')[1])
        if loss <= min_loss:
            best_model = filename
            min_loss = loss
    best_model_filename = os.path.join('out', run_dir, best_model)
    print(best_model_filename)
    return best_model_filename

