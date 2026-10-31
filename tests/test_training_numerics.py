"""Regression tests for SIoU gradients and rotated previous-frame targets."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/lib')]
from dataset.generic_dataset import GenericDataset
from model.losses import IOUloss, SIoU
from lib.utils.image import draw_umich_gaussian, gaussian_radius


class LegacySIoU(SIoU):
    """Preserve the previous angle expression to compare ordinary inputs."""
    def get_angle(self, pred, target):
        pred_ct = (pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2
        target_ct = (target[:, 0] + target[:, 2]) / 2, (target[:, 1] + target[:, 3]) / 2
        dx = torch.abs(target_ct[0] - pred_ct[0])
        dy = torch.abs(target_ct[1] - pred_ct[1])
        distance = torch.sqrt(dx.square() + dy.square())
        x, y = dx / (distance + 1e-16), dy / (distance + 1e-16)
        return pred_ct, target_ct, torch.where(x > math.sqrt(0.5), y, x)


class TrainingNumericsTests(unittest.TestCase):
    def test_siou_coincident_centers_have_finite_gradients(self):
        cases = [
            ([0., 0., 10., 10.], [0., 0., 10., 10.]),
            ([1., 1., 9., 9.], [0., 0., 10., 10.]),
            ([-2., -5., 2., 5.], [-4., -3., 4., 3.]),
            ([0., 0., 0., 0.], [0., 0., 0., 0.]),
        ]
        for dtype in (torch.float32, torch.float64):
            for prediction, truth in cases:
                with self.subTest(dtype=dtype, prediction=prediction, truth=truth):
                    pred = torch.tensor([prediction], dtype=dtype, requires_grad=True)
                    target = torch.tensor([truth], dtype=dtype, requires_grad=True)
                    loss = SIoU()(pred, target).sum()
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(pred.grad).all())
                    self.assertTrue(torch.isfinite(target.grad).all())
                    if prediction == truth and prediction[0] < prediction[2]:
                        self.assertEqual(loss.item(), 0.)

    def test_siou_ordinary_values_and_gradients_match_previous_formula(self):
        pred_values = [[0., 0., 10., 10.], [3., 2., 13., 8.], [-2., 0., 5., 9.]]
        target = torch.tensor([[1., 2., 9., 11.], [1., 1., 14., 11.],
                               [8., 12., 14., 20.]], dtype=torch.float64)
        pred = torch.tensor(pred_values, dtype=torch.float64, requires_grad=True)
        reference = pred.detach().clone().requires_grad_()
        actual, expected = SIoU()(pred, target), LegacySIoU()(reference, target)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(pred.grad, reference.grad, rtol=1e-12, atol=1e-12)

    def test_dense_iou_loss_backward_with_exact_match_and_empty_frame(self):
        # Exercise the actual gather/box reconstruction path used in training.
        output = torch.zeros(2, 4, 2, 2, requires_grad=True)
        target = torch.zeros(2, 1, 4)
        mask = torch.zeros_like(target)
        target[0, 0] = torch.tensor([0., 0., 2., 2.])
        mask[0, 0] = 1
        with torch.no_grad():
            output[0, :, 0, 0] = target[0, 0]
        loss = IOUloss(SimpleNamespace(use_siou=True))(
            output, mask, torch.zeros(2, 1, dtype=torch.long), target,
            torch.zeros(2, 1, 2, 2))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertEqual(output.grad[1].abs().sum().item(), 0.)

    def dataset(self):
        dataset = GenericDataset()
        dataset.opt = SimpleNamespace(
            input_h=128, input_w=128, output_h=32, output_w=32, down_ratio=4,
            pre_hm=True, num_classes=1, hm_disturb=0., lost_disturb=0.,
            fp_disturb=0., dense_reg=1,
            heads={'hm': 1, 'reg': 2, 'tracking': 2, 'ltrb_amodal': 4, 'reid': 128})
        dataset.cat_ids = {1: 1}
        dataset.max_objs = 8
        return dataset

    @staticmethod
    def transformed_box(box, transform):
        # Independent rectangle formula: transformed center +/- |A| times half-size.
        x, y, w, h = box
        center = transform @ np.array([x + w / 2, y + h / 2, 1.])
        extent = np.abs(transform[:, :2]) @ np.array([w / 2, h / 2])
        return np.concatenate((center - extent, center + extent)).astype(np.float32)

    def test_previous_heatmaps_and_motion_keep_rotated_track_identities(self):
        dataset = self.dataset()
        previous = [
            {'bbox': [36., 36., 8., 40.], 'category_id': 1, 'track_id': 21},
            {'bbox': [64., 68., 40., 8.], 'category_id': 1, 'track_id': 42},
        ]
        # Reverse the object order, and include a new identity without history.
        current = [
            {'bbox': [60., 64., 40., 8.], 'category_id': 1, 'track_id': 42},
            {'bbox': [40., 40., 8., 40.], 'category_id': 1, 'track_id': 21},
            {'bbox': [50., 50., 12., 12.], 'category_id': 1, 'track_id': 99},
        ]
        for angle in (0., -30., 30.):
            with self.subTest(angle=angle):
                transform = cv2.getRotationMatrix2D((64., 64.), angle, 1.)
                heatmap, centers, identities = dataset._get_pre_dets(
                    previous, transform, transform / 4)
                self.assertEqual(identities, [21, 42])
                expected_hm = np.zeros_like(heatmap)
                expected_centers = {}
                for ann in previous:
                    box = self.transformed_box(ann['bbox'], transform)
                    center = (box[:2] + box[2:]) / 2
                    expected_centers[ann['track_id']] = center / 4
                    h, w = box[3] - box[1], box[2] - box[0]
                    radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
                    draw_umich_gaussian(expected_hm[0], center.astype(np.int32), radius)
                np.testing.assert_array_equal(heatmap, expected_hm)
                for index, identity in enumerate(identities):
                    np.testing.assert_allclose(centers[index], expected_centers[identity], atol=1e-6)
                ret = {}
                dataset._init_ret(ret)
                for index, ann in enumerate(current):
                    bbox, amodal = dataset._get_bbox_output(ann['bbox'], transform / 4)
                    dataset._add_instance(ret, index, 1, bbox, amodal, ann, centers, identities)
                    self.assertEqual(ret['reid'][index], ann['track_id'])
                    if ann['track_id'] in expected_centers:
                        box = self.transformed_box(ann['bbox'], transform / 4)
                        cell = ((box[:2] + box[2:]) / 2).astype(np.int32)
                        expected_motion = expected_centers[ann['track_id']] - cell
                        np.testing.assert_allclose(ret['tracking'][index], expected_motion, atol=1e-6)
                        np.testing.assert_array_equal(ret['tracking_mask'][index], [1., 1.])
                    else:
                        np.testing.assert_array_equal(ret['tracking_mask'][index], [0., 0.])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
