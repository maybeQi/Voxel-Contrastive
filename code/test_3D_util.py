import math

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=18, stride_z=4, test_save_path=None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))   #metric 即翻译为指标
    metric_dice = []
    metric_jac = []
    metric_hd = []
    metric_asd = []
    print("Testing begin")
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path in image_list:
            ids = image_path.split("/")[-1].replace(".h5", "")   #把文件名的后缀去掉了
            print(ids)
            h5f = h5py.File(image_path, 'r')         #读取出文件放到，h5f
            image = h5f['image'][:]
            label = h5f['label'][:]
            prediction = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            metric = calculate_metric_percase(label == 1, prediction == 1)
            print(metric)
            total_metric[0, :] += metric
            metric_dice.append(metric[0])
            metric_jac.append(metric[1])
            metric_hd.append(metric[2])
            metric_asd.append(metric[3])
            f.writelines("{},{},{},{},{}\n".format(
                ids, metric[0], metric[1], metric[2], metric[3]))

            # prediction[prediction == 1] = -19
            pred_itk = prediction.astype(np.uint8)
            pred_itk = sitk.GetImageFromArray(pred_itk)   #SimplrITK库
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(ids))

            # 筛选边界负样本
            ambiguous_anchors = np.where((label == prediction)&(label == 1), 20, 0)
            ambiguous_anchors = label - ambiguous_anchors
            # ambiguous_anchors = ambiguous_anchors.astype(np.uint8)

            ambiguous_anchors1 = np.where((label == prediction)&(prediction == 1), 80, 0)
            ambiguous_anchors1 = prediction - ambiguous_anchors1
            ambiguous_anchors = ambiguous_anchors -ambiguous_anchors1
            ambiguous_anchors = ambiguous_anchors.astype(np.uint8)

            ambi_anchor_itk = sitk.GetImageFromArray(ambiguous_anchors)
            ambi_anchor_itk.SetSpacing((1.0,1.0,1.0))
            sitk.WriteImage(ambi_anchor_itk,test_save_path +
                            "/{}_ambi.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
        f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
    f.close()
    print("Testing end")
    average = total_metric / len(image_list)
    std = [np.std(metric_dice), np.std(metric_jac), np.std(metric_hd), np.std(metric_asd)]
    std_error = [np.std(metric_dice) / math.sqrt(len(image_list)), np.std(metric_jac)/ math.sqrt(len(image_list)), np.std(metric_hd)/ math.sqrt(len(image_list)), np.std(metric_asd)/ math.sqrt(len(image_list))]
    return average, std


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jaccard = metric.binary.jc(pred, gt)
    ravd = abs(metric.binary.ravd(pred, gt))
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return np.array([dice, jaccard, hd, asd])
