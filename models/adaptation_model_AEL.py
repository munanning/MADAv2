import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplab import DeepLab
from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from schedulers import get_scheduler
from .utils import normalisation_pooling, dequeue_and_enqueue, label_onehot, \
    generate_cutmix_mask, update_cutmix_bank, cal_category_confidence, generate_unsup_data
from loss.loss import get_criterion


class CustomModel:
    def __init__(self, cfg, writer, logger):
        # super(CustomModel, self).__init__()
        self.cfg = cfg
        self.writer = writer
        self.class_numbers = 19
        self.logger = logger
        cfg_model = cfg['model']
        self.cfg_model = cfg_model
        self.best_iou = -100
        self.iter = 0
        self.nets = []
        self.split_gpu = 0
        self.default_gpu = cfg['model']['default_gpu']
        self.PredNet_Dir = None
        self.valid_classes = cfg['training']['valid_classes']
        self.G_train = True
        self.cls_feature_weight = cfg['training']['cls_feature_weight']
        self.centroids = np.zeros((10, 19, 256)).astype('float32')

        bn = cfg_model['bn']
        if bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        elif bn == 'gn':
            BatchNorm = nn.GroupNorm
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(bn))
        self.PredNet = DeepLab(
            num_classes=19,
            backbone=cfg_model['basenet']['version'],
            output_stride=16,
            bn=cfg_model['bn'],
            freeze_bn=True,
        ).cuda()
        self.load_PredNet(cfg, writer, logger, dir=None, net=self.PredNet)
        self.PredNet_DP = self.init_device(self.PredNet, gpu_id=self.default_gpu, whether_DP=True)
        self.PredNet.eval()
        self.PredNet_num = 0

        self.BaseNet = DeepLab(
            num_classes=19,
            backbone=cfg_model['basenet']['version'],
            output_stride=16,
            bn=cfg_model['bn'],
            freeze_bn=False,
        )

        logger.info('the backbone is {}'.format(cfg_model['basenet']['version']))

        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)
        self.nets.extend([self.BaseNet])
        self.nets_DP = [self.BaseNet_DP]

        self.optimizers = []
        self.schedulers = []
        # optimizer_cls = get_optimizer(cfg)
        optimizer_cls = torch.optim.SGD
        optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                            if k != 'name'}
        # optimizer_cls_D = torch.optim.SGD
        # optimizer_params_D = {k:v for k, v in cfg['training']['optimizer_D'].items()
        #                     if k != 'name'}
        # self.BaseOpti = optimizer_cls(self.BaseNet.parameters(), **optimizer_params)
        self.BaseOpti = optimizer_cls(self.BaseNet.optim_parameters(cfg['training']['optimizer']['lr']),
                                      **optimizer_params)
        self.optimizers.extend([self.BaseOpti])

        self.BaseSchedule = get_scheduler(self.BaseOpti, cfg['training']['lr_schedule'])
        self.schedulers.extend([self.BaseSchedule])
        self.setup(cfg, writer, logger)

        self.adv_source_label = 0
        self.adv_target_label = 1
        self.bceloss = nn.BCEWithLogitsLoss(size_average=True)
        # self.loss_fn = get_loss_function(cfg)
        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.smoothloss = nn.SmoothL1Loss()
        self.triplet_loss = nn.TripletMarginLoss()

        # U2PL parameters
        # build class-wise memory bank
        self.memobank = []
        self.queue_ptrlis = []
        self.queue_size = []
        for i in range(cfg["data"]["n_class"]):
            self.memobank.append([torch.zeros(0, 256)])
            self.queue_size.append(30000)
            self.queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
        self.queue_size[0] = 50000

        # build prototype
        self.prototype = torch.zeros(
            (
                cfg["data"]["n_class"],
                cfg["U2PL"]["contrastive"]["num_queries"],
                1,
                256,
            )
        ).cuda()

        # AEL parameters
        self.acp = cfg['AEL'].get('acp', False)
        self.acm = cfg['AEL'].get('acm', False)
        criterion = get_criterion(cfg)
        cons = cfg["AEL"]["criterion"].get("cons", False)

        self.sample = False
        if cons:
            self.sample = cfg["AEL"]["criterion"]["cons"].get("sample", False)
            print('sample == True')
        if cons:
            criterion_cons = get_criterion(cfg, cons=True)
        else:
            criterion_cons = torch.nn.CrossEntropyLoss(ignore_index=255)

        if self.acp or self.acm or self.sample:
            class_criterion = (
                torch.rand(3, self.class_numbers).type(torch.float32).cuda()
            )
        if self.acm:
            cutmix_bank = torch.zeros(
                self.class_numbers, 2975  # target_train_loader.dataset.__len__()
            ).cuda()

        self.criterion = criterion
        self.criterion_cons = criterion_cons
        self.class_criterion = class_criterion
        self.cutmix_bank = cutmix_bank
        self.threshold = cfg["AEL"]["criterion"].get("threshold", 0)
        self.consist_weight = cfg["AEL"]["criterion"].get("consist_weight", 1)
        self.contra_weight = cfg["AEL"]["criterion"].get("contra_weight", 1)

        # acp
        if self.acp:
            self.cfg_acp = cfg['AEL']['acp']
            all_cat = [i for i in range(self.class_numbers)]
            ignore_cat = [0, 1, 2, 8, 10]
            self.target_cat = list(set(all_cat) - set(ignore_cat))
            self.class_momentum = cfg['AEL']['acp'].get("momentum", 0.999)
            self.num_cat = cfg['AEL']['acp'].get("number", 3)

        # acm
        if self.acm:
            self.class_momentum = cfg['AEL']["acm"].get("momentum", 0.999)
            self.area_thresh = cfg['AEL']["acm"].get("area_thresh", 0.0001)
            self.no_pad = cfg['AEL']["acm"].get("no_pad", False)
            self.no_slim = cfg['AEL']["acm"].get("no_slim", False)
            if "area_thresh2" in cfg['AEL']["acm"].keys():
                self.area_thresh2 = cfg['AEL']["acm"]["area_thresh2"]
            else:
                self.area_thresh2 = self.area_thresh

    def create_PredNet(self, ):
        ss = DeepLab(
            num_classes=19,
            backbone=self.cfg_model['basenet']['version'],
            output_stride=16,
            bn=self.cfg_model['bn'],
            freeze_bn=True,
        ).cuda()
        ss.eval()
        return ss

    def setup(self, cfg, writer, logger):
        """
        set optimizer and load pretrained model
        """
        for net in self.nets:
            # name = net.__class__.__name__
            self.init_weights(cfg['model']['init'], logger, net)
            print("Initializition completed")
            if hasattr(net, '_load_pretrained_model') and cfg['model']['pretrained']:
                print("loading pretrained model for {}".format(net.__class__.__name__))
                net._load_pretrained_model()
        '''load pretrained model'''
        if cfg['training']['resume_flag']:
            self.load_nets(cfg, writer, logger)
        pass

    def forward(self, input):
        feat, feat_low, feat_cls, output = self.BaseNet_DP(input)
        return feat, feat_low, feat_cls, output

    def forward_Up(self, input):
        feat, feat_low, feat_cls, output = self.forward(input)
        output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)
        return feat, feat_low, feat_cls, output

    def PredNet_Forward(self, input):
        with torch.no_grad():
            _, _, feat_cls, output_result = self.PredNet_DP(input)
        return _, _, feat_cls, output_result

    def calculate_mean_vector(self, feat_cls, outputs, labels, ):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        labels_expanded = self.process_label(labels)
        outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                scale = torch.sum(outputs_pred[n][t]) / labels.shape[2] / labels.shape[3] * 2
                s = normalisation_pooling()(s, scale)
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def step_active_stage2(self, epoch, images_unsup_weak, images_sup, labels_sup,
                           sample_cat, img_id, sample_id, percent):
        # student model forward
        batch_size, c, h, w = images_sup.size()
        _, _, reps_student_sup, pred_student_sup = self.forward(images_sup)
        batch_size, c, h_small, w_small = pred_student_sup.size()

        preds_student_sup = F.interpolate(pred_student_sup, (h, w), mode="bilinear", align_corners=True)
        loss_sup_student = self.criterion(preds_student_sup, labels_sup)

        # teacher model forward
        with torch.no_grad():
            _, _, reps_teacher_sup, preds_teacher_sup = self.PredNet_Forward(images_sup)
            reps_teacher_sup = reps_teacher_sup.detach()
            preds_teacher_sup = preds_teacher_sup.detach()
            preds_teacher_sup = F.interpolate(
                preds_teacher_sup, (h, w), mode="bilinear", align_corners=True
            )

            self.PredNet.eval()
            self.PredNet_DP.eval()
            _, _, reps_teacher_unsup, preds_teacher_unsup = self.PredNet_Forward(images_unsup_weak)
            preds_teacher_unsup = preds_teacher_unsup.detach()
            preds_teacher_unsup = F.interpolate(
                preds_teacher_unsup, (h, w), mode="bilinear", align_corners=True
            )
            if self.acm:
                valid_mask_mix = generate_cutmix_mask(
                    preds_teacher_unsup[1].max(0)[1].cpu().numpy(),
                    sample_cat,
                    self.area_thresh,
                    no_pad=self.no_pad,
                    no_slim=self.no_slim,
                )

                # update cutmix bank
                self.cutmix_bank = update_cutmix_bank(
                    self.cutmix_bank, preds_teacher_unsup, img_id, sample_id, self.area_thresh2
                )

                images_unsup_strong, preds_teacher_unsup = generate_unsup_data(
                    images_unsup_weak, preds_teacher_unsup, valid_mask_mix)

            # compute consistency loss
            logits_teacher_sup = preds_teacher_sup.max(1)[1]
            conf_sup = F.softmax(preds_teacher_sup, dim=1).max(1)[0]
            conf_teacher_sup_map = conf_sup
            logits_teacher_sup[conf_teacher_sup_map < self.threshold] = 255

            probs_teacher_unsup = F.softmax(preds_teacher_unsup, dim=1)
            entropy_teacher_unsup = -torch.sum(
                probs_teacher_unsup * torch.log(probs_teacher_unsup + 1e-10), dim=1
            )
            thresh = np.percentile(
                entropy_teacher_unsup.detach().cpu().numpy().flatten(), percent
            )
            conf_unsup = F.softmax(preds_teacher_unsup, dim=1).max(1)[0]
            logits_teacher_unsup = preds_teacher_unsup.max(1)[1]

            logits_teacher_unsup[entropy_teacher_unsup < thresh] = 255

            self.PredNet.train()
            self.PredNet_DP.train()
            _, _, reps_teacher_unsup, _ = self.PredNet_Forward(images_unsup_strong)
            reps_teacher_unsup = reps_teacher_unsup.detach()
            prob_l_teacher = F.softmax(
                F.interpolate(
                    preds_teacher_sup,
                    (h_small, w_small),
                    mode="bilinear",
                    align_corners=True,
                ),
                dim=1,
            ).detach()
            prob_u_teacher = F.softmax(
                F.interpolate(
                    preds_teacher_unsup,
                    (h_small, w_small),
                    mode="bilinear",
                    align_corners=True,
                ),
                dim=1,
            ).detach()

        _, _, reps_student_unsup, preds_student_unsup = self.forward(images_unsup_strong)
        preds_student_unsup = F.interpolate(preds_student_unsup, (h, w), mode="bilinear", align_corners=True)

        # consistency loss
        with torch.no_grad():
            if self.acp or self.acm:
                category_entropy = cal_category_confidence(
                    preds_student_sup.detach(),
                    preds_student_unsup.detach(),
                    labels_sup,
                    preds_teacher_unsup,
                    self.class_numbers
                )
                # perform momentum update
                class_criterion = (
                        self.class_criterion * self.class_momentum
                        + category_entropy.cuda() * (1 - self.class_momentum)
                )
        if isinstance(self.criterion_cons, torch.nn.CrossEntropyLoss):
            loss_consistency1 = (
                self.criterion_cons(preds_student_sup, logits_teacher_sup)
            )
            loss_consistency2 = (
                self.criterion_cons(preds_student_unsup, logits_teacher_unsup)
            )

        elif self.sample:
            loss_consistency1 = (
                self.criterion_cons(
                    preds_student_sup,
                    conf_sup,
                    logits_teacher_sup,
                    class_criterion[0],
                )
            )
            loss_consistency2 = (
                self.criterion_cons(
                    preds_student_unsup,
                    conf_unsup,
                    logits_teacher_unsup,
                    class_criterion[0],
                )
            )

        else:
            loss_consistency1 = (
                self.criterion_cons(preds_student_sup, conf_sup, logits_teacher_sup)
            )
            loss_consistency2 = (
                self.criterion_cons(preds_student_unsup, conf_unsup, logits_teacher_unsup)
            )

        loss_consistency = loss_consistency1 + loss_consistency2

        if epoch > 50:
            # distance loss
            _, _, target_feat_cls, target_output = self.forward(images_unsup_weak)
            target_outputs_softmax = F.softmax(target_output, dim=1)
            target_outputs_argmax = target_outputs_softmax.argmax(dim=1, keepdim=True)
            target_vectors, target_ids = self.calculate_mean_vector(target_feat_cls, target_output,
                                                                    target_outputs_argmax.float())
            target_objective_vectors = torch.zeros([19, 256]).cuda()
            for t in range(len(target_ids)):
                target_objective_vectors[target_ids[t]] = target_vectors[t].squeeze()
            # calculate min distance
            loss_L2_target_cls, min_index = self.calculate_min_mse(target_objective_vectors)

            # update features
            self.centroids[min_index] = self.centroids[min_index] * 0.999 + \
                                        0.001 * target_objective_vectors.detach().cpu().numpy()
        else:
            loss_L2_target_cls = 0 * pred_student_sup.sum()

        loss = (
                loss_sup_student
                + self.consist_weight * loss_consistency
                + self.cls_feature_weight * loss_L2_target_cls
        )

        self.BaseOpti.zero_grad()
        loss.backward()
        self.BaseOpti.step()
        return loss, loss_consistency.item(), loss_sup_student.item(), loss_L2_target_cls.item()

    def step_active_stage2_EMA(self, epoch, images_unsup_weak, images_sup, labels_sup,
                               sample_cat, img_id, sample_id, percent):
        # student model forward
        batch_size, c, h, w = images_sup.size()
        _, _, reps_student_sup, pred_student_sup = self.forward(images_sup)
        batch_size, c, h_small, w_small = pred_student_sup.size()

        preds_student_sup = F.interpolate(pred_student_sup, (h, w), mode="bilinear", align_corners=True)
        loss_sup_student = self.criterion(preds_student_sup, labels_sup)

        # teacher model forward
        with torch.no_grad():
            _, _, reps_teacher_sup, preds_teacher_sup = self.PredNet_Forward(images_sup)
            reps_teacher_sup = reps_teacher_sup.detach()
            preds_teacher_sup = preds_teacher_sup.detach()
            preds_teacher_sup = F.interpolate(
                preds_teacher_sup, (h, w), mode="bilinear", align_corners=True
            )

            self.PredNet.eval()
            self.PredNet_DP.eval()
            _, _, reps_teacher_unsup, preds_teacher_unsup = self.PredNet_Forward(images_unsup_weak)
            preds_teacher_unsup = preds_teacher_unsup.detach()
            preds_teacher_unsup = F.interpolate(
                preds_teacher_unsup, (h, w), mode="bilinear", align_corners=True
            )
            if self.acm:
                valid_mask_mix = generate_cutmix_mask(
                    preds_teacher_unsup[1].max(0)[1].cpu().numpy(),
                    sample_cat,
                    self.area_thresh,
                    no_pad=self.no_pad,
                    no_slim=self.no_slim,
                )

                # update cutmix bank
                self.cutmix_bank = update_cutmix_bank(
                    self.cutmix_bank, preds_teacher_unsup, img_id, sample_id, self.area_thresh2
                )

                images_unsup_strong, preds_teacher_unsup = generate_unsup_data(
                    images_unsup_weak, preds_teacher_unsup, valid_mask_mix)

            # compute consistency loss
            logits_teacher_sup = preds_teacher_sup.max(1)[1]
            conf_sup = F.softmax(preds_teacher_sup, dim=1).max(1)[0]
            conf_teacher_sup_map = conf_sup
            logits_teacher_sup[conf_teacher_sup_map < self.threshold] = 255

            probs_teacher_unsup = F.softmax(preds_teacher_unsup, dim=1)
            entropy_teacher_unsup = -torch.sum(
                probs_teacher_unsup * torch.log(probs_teacher_unsup + 1e-10), dim=1
            )
            thresh = np.percentile(
                entropy_teacher_unsup.detach().cpu().numpy().flatten(), percent
            )
            conf_unsup = F.softmax(preds_teacher_unsup, dim=1).max(1)[0]
            logits_teacher_unsup = preds_teacher_unsup.max(1)[1]

            logits_teacher_unsup[entropy_teacher_unsup < thresh] = 255

            self.PredNet.train()
            self.PredNet_DP.train()
            _, _, reps_teacher_unsup, _ = self.PredNet_Forward(images_unsup_strong)
            reps_teacher_unsup = reps_teacher_unsup.detach()
            prob_l_teacher = F.softmax(
                F.interpolate(
                    preds_teacher_sup,
                    (h_small, w_small),
                    mode="bilinear",
                    align_corners=True,
                ),
                dim=1,
            ).detach()
            prob_u_teacher = F.softmax(
                F.interpolate(
                    preds_teacher_unsup,
                    (h_small, w_small),
                    mode="bilinear",
                    align_corners=True,
                ),
                dim=1,
            ).detach()

        _, _, reps_student_unsup, preds_student_unsup = self.forward(images_unsup_strong)
        preds_student_unsup = F.interpolate(preds_student_unsup, (h, w), mode="bilinear", align_corners=True)

        # consistency loss
        with torch.no_grad():
            if self.acp or self.acm:
                category_entropy = cal_category_confidence(
                    preds_student_sup.detach(),
                    preds_student_unsup.detach(),
                    labels_sup,
                    preds_teacher_unsup,
                    self.class_numbers
                )
                # perform momentum update
                class_criterion = (
                        self.class_criterion * self.class_momentum
                        + category_entropy.cuda() * (1 - self.class_momentum)
                )
        if isinstance(self.criterion_cons, torch.nn.CrossEntropyLoss):
            loss_consistency1 = (
                self.criterion_cons(preds_student_sup, logits_teacher_sup)
            )
            loss_consistency2 = (
                self.criterion_cons(preds_student_unsup, logits_teacher_unsup)
            )

        elif self.sample:
            loss_consistency1 = (
                self.criterion_cons(
                    preds_student_sup,
                    conf_sup,
                    logits_teacher_sup,
                    class_criterion[0],
                )
            )
            loss_consistency2 = (
                self.criterion_cons(
                    preds_student_unsup,
                    conf_unsup,
                    logits_teacher_unsup,
                    class_criterion[0],
                )
            )

        else:
            loss_consistency1 = (
                self.criterion_cons(preds_student_sup, conf_sup, logits_teacher_sup)
            )
            loss_consistency2 = (
                self.criterion_cons(preds_student_unsup, conf_unsup, logits_teacher_unsup)
            )

        loss_consistency = loss_consistency1 + loss_consistency2

        if epoch > 50:
            # distance loss
            _, _, target_feat_cls, target_output = self.forward(images_unsup_weak)
            target_outputs_softmax = F.softmax(target_output, dim=1)
            target_outputs_argmax = target_outputs_softmax.argmax(dim=1, keepdim=True)
            target_vectors, target_ids = self.calculate_mean_vector(target_feat_cls, target_output,
                                                                    target_outputs_argmax.float())
            target_objective_vectors = torch.zeros([19, 256]).cuda()
            for t in range(len(target_ids)):
                target_objective_vectors[target_ids[t]] = target_vectors[t].squeeze()
            # calculate min distance
            loss_L2_target_cls, min_index = self.calculate_min_mse(target_objective_vectors)

            # update features
            self.centroids[min_index] = self.centroids[min_index] * 0.999 + \
                                        0.001 * target_objective_vectors.detach().cpu().numpy()
        else:
            loss_L2_target_cls = 0 * pred_student_sup.sum()

        loss = (
                loss_sup_student
                + self.consist_weight * loss_consistency
                + self.cls_feature_weight * loss_L2_target_cls
        )

        self.BaseOpti.zero_grad()
        loss.backward()
        self.BaseOpti.step()

        # update ema model
        for param_q, param_k in zip(self.BaseNet.parameters(), self.PredNet.parameters()):
            param_k.data = param_k.data.clone() * 0.999 + param_q.data.clone() * (1. - 0.999)
        for buffer_q, buffer_k in zip(self.BaseNet.buffers(), self.PredNet.buffers()):
            buffer_k.data = buffer_q.data.clone()

        return loss, loss_consistency.item(), loss_sup_student.item(), loss_L2_target_cls.item()

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, 20, w, h).cuda()
        id = torch.where(label < 19, label, torch.Tensor([19]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def calculate_min_mse(self, single_image_objective_vectors):
        loss = []
        for centroid in self.centroids:
            new_loss = torch.mean((single_image_objective_vectors - torch.Tensor(centroid).cuda()) ** 2)
            loss.append(new_loss)

        min_loss = min(loss)
        min_index = loss.index(min_loss)

        sum_loss = sum(loss)
        weights = []
        weighted_loss = []
        for item in loss:
            weight = 1 / item
            weighted_loss.append(weight * item)
            weights.append(weight)
        return sum(weighted_loss) / sum(weights), min_index

    def scheduler_step(self):
        # for net in self.nets:
        #     self.schedulers[net.__class__.__name__].step()
        for scheduler in self.schedulers:
            scheduler.step()

    def optimizer_zerograd(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].zero_grad()
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def optimizer_step(self):
        # for net in self.nets:
        #     self.optimizers[net.__class__.__name__].step()
        for opt in self.optimizers:
            opt.step()

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net

    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger != None:
                logger.info("Successfully set the model eval mode")
        else:
            net.eval()
            if logger != None:
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net == None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
            # if logger!=None:
            #     logger.info("Successfully set the model train mode")
        else:
            net.train()
            # if logger!= None:
            #     logger.info(print("Successfully set {} train mode".format(net.__class__.__name__)))
        return

    def init_weights(self, cfg, logger, net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        init_type = cfg.get('init_type', init_type)
        init_gain = cfg.get('init_gain', init_gain)

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, SynchronizedBatchNorm2d) or classname.find('BatchNorm2d') != -1 \
                    or isinstance(m, nn.GroupNorm):
                # or isinstance(m, InPlaceABN) or isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.

        print('initialize {} with {}'.format(init_type, net.__class__.__name__))
        logger.info('initialize {} with {}'.format(init_type, net.__class__.__name__))
        net.apply(init_func)  # apply the initialization function <init_func>
        pass

    def adaptive_load_nets(self, net, model_weight):
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in model_weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    def load_nets(self, cfg, writer, logger):  # load pretrained weights on the net
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            _k = -1
            for net in self.nets:
                name = net.__class__.__name__
                _k += 1
                if checkpoint.get(name) == None:
                    continue
                if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                    continue
                self.adaptive_load_nets(net, checkpoint[name]["model_state"])
                if cfg['training']['optimizer_resume']:
                    self.adaptive_load_nets(self.optimizers[_k], checkpoint[name]["optimizer_state"])
                    self.adaptive_load_nets(self.schedulers[_k], checkpoint[name]["scheduler_state"])
            self.iter = checkpoint["iter"]
            self.best_iou = checkpoint['best_iou']
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["iter"]
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(cfg['training']['resume']))

    def load_PredNet(self, cfg, writer, logger, dir=None, net=None):  # load pretrained weights on the net
        dir = dir or cfg['training']['Pred_resume']
        best_iou = 0
        if os.path.isfile(dir):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(dir)
            )
            checkpoint = torch.load(dir)
            name = net.__class__.__name__
            if checkpoint.get(name) == None:
                return
            if name.find('FCDiscriminator') != -1 and cfg['training']['gan_resume'] == False:
                return
            self.adaptive_load_nets(net, checkpoint[name]["model_state"])
            iter = checkpoint["iter"]
            best_iou = checkpoint['best_iou']
            logger.info(
                "Loaded checkpoint '{}' (iter {}) (best iou {}) for PredNet".format(
                    dir, checkpoint["iter"], best_iou
                )
            )
        else:
            raise Exception("No checkpoint found at '{}'".format(dir))
        if hasattr(net, 'best_iou'):
            net.best_iou = best_iou
        return best_iou
