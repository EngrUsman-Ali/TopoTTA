import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred_flat = pred_mask.astype(np.uint8).ravel()
    gt_flat = gt_mask.astype(np.uint8).ravel()
    tp = int(np.logical_and(pred_mask, gt_mask).sum())
    fp = int(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
    fn = int(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())

    precision = float(precision_score(gt_flat, pred_flat, zero_division=0))
    recall = float(recall_score(gt_flat, pred_flat, zero_division=0))
    f1 = float(f1_score(gt_flat, pred_flat, zero_division=0))
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}

def mean(list_of_metric_dicts):
    if not list_of_metric_dicts:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
    precision = float(np.mean([m['precision'] for m in list_of_metric_dicts]))
    recall = float(np.mean([m['recall'] for m in list_of_metric_dicts]))
    f1 = float(np.mean([m['f1'] for m in list_of_metric_dicts]))
    iou = float(np.mean([m['iou'] for m in list_of_metric_dicts]))
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}
