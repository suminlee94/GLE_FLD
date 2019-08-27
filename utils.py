import torch
import torch.nn as nn
import numpy as np

def cal_loss(sample, output):
    batch_size, _, pred_w, pred_h = sample['image'].size()

    lm_size = int(output['lm_pos_map'].size(2))
    visibility = sample['landmark_vis']
    vis_mask = torch.cat([visibility.reshape(batch_size* 8, -1)] * lm_size * lm_size, dim=1).float()
    lm_map_gt = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
    lm_pos_map = output['lm_pos_map']
    lm_map_pred =lm_pos_map.reshape(batch_size * 8, -1)

    loss = torch.pow(vis_mask * (lm_map_pred - lm_map_gt), 2).mean()

    return loss

class Evaluator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def add(self, output, sample):
        landmark_vis_count = sample['landmark_vis'].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample['landmark_vis'].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        lm_pos_map = output['lm_pos_map']
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)

        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2).cpu().numpy(), (pred_h, pred_w))
        lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)

        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * lm_pos_output - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def evaluate(self):
        lm_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist_all = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {'lm_dist' : lm_dist,
                'lm_dist_all' : lm_dist_all}
