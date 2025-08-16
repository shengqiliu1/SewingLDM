import torch
import torch.nn.functional as F
from torch import nn
import time
import torchvision
from sklearn.metrics import confusion_matrix, classification_report


from diffusion.utils.composed_loss import ComposedLoss, ComposedPatternLoss
from diffusion.model.utils import set_grad_checkpoint

class StitchLoss():
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, similarity_matrix, gt_pos_neg_indices):
        simi_matrix = similarity_matrix.reshape(-1, similarity_matrix.shape[-1])
        tmp = simi_matrix[gt_pos_neg_indices[:, :, 0]]
        simi_res = torch.gather(tmp, -1, gt_pos_neg_indices[:, :, 1].unsqueeze(-1)) / 0.01
        ce_label = torch.zeros(simi_res.shape[0]).to(simi_res.device)
        return F.cross_entropy(simi_res.squeeze(-1), ce_label.long()), (torch.max(simi_res.squeeze(-1), 1)[1] == 0).sum() * 1.0 / simi_res.shape[0]

class StitchSimpleLoss():
    def __call__(self, similarity_matrix, gt_matrix, gt_free_mask=None):
        if gt_free_mask is not None:
            gt_free_mask = gt_free_mask.reshape(gt_free_mask.shape[0], -1)
            similarity_matrix = torch.masked_fill(similarity_matrix, gt_free_mask.unsqueeze(1), -float("inf"))
            similarity_matrix = torch.masked_fill(similarity_matrix, gt_free_mask.unsqueeze(-1), 0)

        simi_matrix = (similarity_matrix / 0.01).reshape(-1, similarity_matrix.shape[-1])
        gt = gt_matrix.reshape(-1, gt_matrix.shape[-1])
        gt_labels = torch.argmax(gt, dim=1).long()
        return F.nll_loss(F.log_softmax(simi_matrix, dim=-1), gt_labels), (torch.argmax(simi_matrix, dim=1) == gt_labels).sum() / simi_matrix.shape[0]



class SetCriterionWithOutMatcher(nn.Module):

    def __init__(self, data_config, in_config={}):
        super().__init__()
        self.config = {}
        self.config['loss'] = {
            'loss_components': ['shape', 'loop', 'rotation', 'translation', 'stitch', 'free_class', 'edge_type'],
            'quality_components': ['shape', 'discrete', 'rotation', 'translation', 'stitch', 'free_class'],
            'panel_origin_invariant_loss': False,
            'loop_loss_weight': 1.,
            'stitch_tags_margin': 0.3,
            'epoch_with_stitches': 10000, 
            'stitch_supervised_weight': 0.1,   # only used when explicit stitch loss is used
            'stitch_hardnet_version': False,
            'panel_origin_invariant_loss': True
        }

        self.config['loss'].update(in_config)

        self.composed_loss = ComposedPatternLoss(data_config, self.config['loss'])
        self.el_size = data_config['element_size']

        self.stitch_cls_loss = nn.BCEWithLogitsLoss()
        self.stitch_ce_loss = StitchLoss()
        self.stitch_simi_loss = StitchSimpleLoss()
    
    def forward(self, outputs, ground_truth, names=None, epoch=1000):

        b, q = outputs["outlines"].shape[0], outputs["rotations"].shape[1]
        outputs["outlines"] = outputs["outlines"].view(b, q, -1, self.el_size).contiguous()
        full_loss, loss_dict, _ = self.composed_loss(outputs, ground_truth, names, epoch)

        if "edge_cls" in outputs and 'lepoch' in self.config['loss'] and epoch >= self.config['loss']['lepoch']:
            if epoch == -1:
                st_edge_precision, st_edge_recall, st_edge_f1_score, st_precision, st_recall, st_f1_score, st_adj_precs, st_adj_recls, st_adj_f1s = self.prediction_stitch_rp(outputs, ground_truth)
                loss_dict.update({"st_edge_prec": st_edge_precision,
                                  "st_edge_recl": st_edge_recall,
                                  "st_edge_f1s": st_edge_f1_score,
                                  "st_prec": st_precision,
                                  "st_recl": st_recall,
                                  "st_f1s": st_f1_score,
                                  "st_adj_precs": st_adj_precs, 
                                  "st_adj_recls": st_adj_recls, 
                                  "st_adj_f1s": st_adj_f1s})
            edge_cls_gt = (~ground_truth["free_edges_mask"].view(b, -1)).float().to(outputs["edge_cls"].device)
            edge_cls_loss = torchvision.ops.sigmoid_focal_loss(outputs["edge_cls"].squeeze(-1), edge_cls_gt, reduction="mean")
            
            if self.config["loss"]["stitches"] == "ce" and epoch != -1:
                full_loss = full_loss * 10
                loss_dict.update({"stitch_cls_loss": 0.5 * edge_cls_loss})
                edge_cls_acc = ((F.sigmoid(outputs["edge_cls"].squeeze(-1)) > 0.5) == edge_cls_gt).sum().float() / (edge_cls_gt.shape[0] * edge_cls_gt.shape[1])
                loss_dict.update({"stitch_edge_cls_acc": edge_cls_acc})
                full_loss += loss_dict["stitch_cls_loss"]
                # ce loss
                stitch_loss, stitch_acc= self.stitch_ce_loss(outputs["edge_similarity"], ground_truth["label_indices"])
                if stitch_loss is not None and stitch_acc is not None:
                    loss_dict.update({"stitch_ce_loss": 0.01 * stitch_loss, "stitch_acc": stitch_acc})
                    full_loss += loss_dict["stitch_ce_loss"]
            elif self.config["loss"]["stitches"] == "simple" or epoch == -1:
                full_loss = full_loss * 5
                loss_dict.update({"stitch_cls_loss": 0.5 * edge_cls_loss})
                edge_cls_acc = ((F.sigmoid(outputs["edge_cls"].squeeze(-1)) > 0.5) == edge_cls_gt).sum().float() / (edge_cls_gt.shape[0] * edge_cls_gt.shape[1])
                loss_dict.update({"stitch_edge_cls_acc": edge_cls_acc})
                full_loss += loss_dict["stitch_cls_loss"]
                # simi loss
                stitch_loss, stitch_acc = self.stitch_simi_loss(outputs["edge_similarity"], ground_truth["stitch_adj"], ground_truth["free_edges_mask"])
                if stitch_loss is not None and stitch_acc is not None:
                    loss_dict.update({"stitch_ce_loss": 0.05 * stitch_loss, "stitch_acc": stitch_acc})
                    full_loss += loss_dict["stitch_ce_loss"]
            else:
                print("No Stitch Loss")
                stitch_loss, stitch_acc = None, None 

            if "smpl_joints" in ground_truth and "smpl_joints" in outputs:
                joints_loss = F.mse_loss(outputs["smpl_joints"], ground_truth["smpl_joints"])
                loss_dict.update({"smpl_joint_loss": joints_loss})
                full_loss += loss_dict["smpl_joint_loss"]
        
        return full_loss, loss_dict
    
    def prediction_stitch_rp(self, outputs, ground_truth):
        # only support batchsize=1
        if "edge_cls" in outputs:
            bs, q = outputs["outlines"].shape[0], outputs["rotations"].shape[1]
            st_edge_pres, st_edge_recls, st_edge_f1s, st_precs, st_recls, st_f1s, st_adj_precs, st_adj_recls, st_adj_f1s = [], [], [], [], [], [], [], [], []
            
            # import pdb; pdb.set_trace()

            for b in range(bs):
                edge_cls_gt = (~ground_truth["free_edges_mask"][b]).flatten()
                edge_cls_pr = (F.sigmoid(outputs["edge_cls"][b].squeeze(-1)) > 0.5).flatten()
                cls_rept = classification_report(edge_cls_gt.detach().cpu().numpy(), edge_cls_pr.detach().cpu().numpy(), labels=[0,1])
                strs = cls_rept.split("\n")[3].split()
                st_edge_precision, st_edge_recall, st_edge_f1_score = float(strs[1]), float(strs[2]), float(strs[3])
                edge_similarity = outputs["edge_similarity"][b]

                st_cls_pr = (F.sigmoid(outputs["edge_similarity"][b].squeeze(-1)) > 0.5).flatten()
                stitch_cls_rept = classification_report(st_cls_pr.detach().cpu().numpy(), ground_truth["stitch_adj"][b].flatten().detach().cpu().numpy(), labels=[0, 1])
                strs = stitch_cls_rept.split("\n")[3].split()
                st_adj_edge_precision, st_adj_edge_recall, st_adj_edge_f1_score = float(strs[1]), float(strs[2]), float(strs[3])

                st_adj_precs.append(st_adj_edge_precision)
                st_adj_recls.append(st_adj_edge_recall)
                st_adj_f1s.append(st_adj_edge_f1_score)

                if self.config["loss"]["stitches"] == "simple":
                    simi_matrix = edge_similarity + edge_similarity.transpose(0, 1)
                simi_matrix = torch.masked_fill(edge_similarity, (~edge_cls_pr).unsqueeze(0), -float("inf"))
                simi_matrix = torch.masked_fill(simi_matrix, (~edge_cls_pr).unsqueeze(-1), 0)
                simi_matrix = torch.triu(simi_matrix, diagonal=1)
                stitches = []
                num_stitches = edge_cls_pr.nonzero().shape[0] // 2
                for i in range(num_stitches):
                    index = (simi_matrix == torch.max(simi_matrix)).nonzero()
                    stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
                    simi_matrix[index[0, 0], :] = -float("inf")
                    simi_matrix[index[0, 1], :] = -float("inf")
                    simi_matrix[:, index[0, 0]] = -float("inf")
                    simi_matrix[:, index[0, 1]] = -float("inf")
                
                st_precision, st_recall, st_f1_score = SetCriterionWithOutMatcher.set_precision_recall(stitches, ground_truth["stitches"][b])

                st_edge_pres.append(st_edge_precision)
                st_edge_recls.append(st_edge_recall)
                st_edge_f1s.append(st_edge_f1_score)
                st_precs.append(st_precision)
                st_recls.append(st_recall)
                st_f1s.append(st_f1_score)
                # print(st_precision, st_recall, st_f1_score)

            return st_edge_pres, st_edge_recls, st_edge_f1s, st_precs, st_recls, st_f1s, st_adj_precs, st_adj_recls, st_adj_f1s
    
    @staticmethod
    def set_precision_recall(pred_stitches, gt_stitches):
        
        def elem_eq(a, b):
            return (a[0] == b[0] and a[1] == b[1]) or (a[0] == b[1] and a[1] == b[0])
        
        gt_stitches = gt_stitches.transpose(0, 1).cpu().detach().numpy()
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for pstitch in pred_stitches:
            for gstitch in gt_stitches:
                if elem_eq(pstitch, gstitch):
                    true_pos += 1
        false_pos = len(pred_stitches) - true_pos
        false_neg = len(gt_stitches) - (gt_stitches == -1).sum() / 2  - true_pos

        precision = true_pos / (true_pos + false_pos + 1e-6)
        recall = true_pos / (true_pos + false_neg)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        return precision, recall, f1_score
    
    def with_quality_eval(self, ):
        if hasattr(self.composed_loss, "with_quality_eval"):
            self.composed_loss.with_quality_eval = True
    
    def without_quality_eval(self, ):
        if hasattr(self.composed_loss, "with_quality_eval"):
            self.composed_loss.with_quality_eval = False
    
    def print_debug(self):
        self.composed_loss.debug_prints = True
    
    def train(self, mode=True):
        super().train(mode)
        self.composed_loss.train(mode)

    def eval(self):
        super().eval()
        if isinstance(self.composed_loss, object):
            self.composed_loss.eval()
            

def build_loss(cfg, train=True):
    criterion = SetCriterionWithOutMatcher(cfg.start_config, cfg.loss)
    if train:
        criterion.train()
    return criterion