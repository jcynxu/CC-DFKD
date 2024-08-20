import datafree
from typing import Generator
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv
from datafree.utils import ImagePool, DataIter, clip_images
from datafree.metrics.loss import SupConLoss
from datafree.metrics.loss import kdloss
import collections
from torchvision import transforms
from kornia import augmentation




class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)

class MultiTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str( self.transform )




class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        feat = feat.to(self.device)
        n, c = feat.shape
        assert self.dim_feat==c and self.max_size % n==0, "%d, %d"%(self.dim_feat, c, self.max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            #return self.data[:self._ptr]
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

def get_sim_criterion(args, device):
    sim_criterion = SupConLoss(temperature=args.temp, device=device)
    return sim_criterion

class Synthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size, 
                 feature_layers=None, bank_size=40960, n_neg=4096, head_dim=128, init_dataset=None,
                 iterations=100, lr_g=0.1, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 save_dir='run/CC-DFKD', transform=None,
                 autocast=None, use_fp16=False, 
                 normalizer=None, device='cpu', distributed=False,dataset=None,
                 co_alpha=1, co_beta=1, co_gamma=1, co_eta=1, temp=0.07, use_amp=False,lr_z=0.01):
        super(Synthesizer, self).__init__(teacher, student)
        ######
        self.dataset = dataset
        self.co_alpha = co_alpha
        self.co_beta = co_beta
        self.co_gamma = co_gamma
        self.co_eta = co_eta
        self.temp = temp
        self.use_amp = use_amp
        self.lr_z = lr_z
        ######
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.progressive_scale = progressive_scale
        self.nz = nz
        self.n_neg = n_neg
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.bank_size = bank_size
        self.init_dataset = init_dataset

        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None

        self.cmi_hooks = []
        if feature_layers is not None:
            for layer in feature_layers:
                self.cmi_hooks.append( InstanceMeanHook(layer) )
        else:
            for m in teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.cmi_hooks.append( InstanceMeanHook(m) )

        with torch.no_grad():
            teacher.eval()
            fake_inputs = torch.randn(size=(1, *img_size), device=device)
            _ = teacher(fake_inputs)
            cmi_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
            print("dims: %d"%(cmi_feature.shape[1]))
            del fake_inputs
        
        self.generator = generator.to(device).train()
        # local and global bank
        self.mem_bank = MemoryBank('cpu', max_size=self.bank_size, dim_feat=2*cmi_feature.shape[1]) # local + global
        
        self.head = MLPHead(cmi_feature.shape[1], head_dim).to(device).train()
        self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_g)

        self.device = device
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

        self.aug = MultiTransform([
            # global view
            transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
        ])



    def synthesize(self, targets=None):
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        
        #inputs = torch.randn( size=(self.synthesis_batch_size, *self.img_size), device=self.device ).requires_grad_()
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_()
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        generator2 = datafree.models.generator.GeneratorD(nz=self.nz, nc=3, img_size=32, num_classes=self.num_classes)
        if torch.cuda.is_available():
            device = torch.device("cuda")  # 定义设备为GPU
            generator2 = generator2.to(device)

        fast_generator = self.generator.clone()

        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])
        inputs = fast_generator(z)
        for it in range(self.iterations):

            #global_view, local_view = self.aug(inputs) # crop and normalize

            #############################################
            # Generation Loss
            #############################################
            with torch.cuda.amp.autocast():
                loss_G = 0
                z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_()
                #z = nn.Parameter(torch.randn((self.sample_batch_size, self.nz, 1, 1)).to(self.device))
                labels = torch.randint(0, self.num_classes, (self.sample_batch_size,), dtype=torch.long).to(self.device)
                optimizer_G = torch.optim.Adam([{'params': generator2.parameters()}, {'params': [z]}], self.lr_g,
                                               betas=[0.5, 0.999])
                optimizer_G.zero_grad()
                generator2.train()
                # generator.apply(fix_bn)
                fake = generator2(z, labels)

                t_fea, t_logit = self.teacher(fake, return_features=True)
                s_fea, s_logit = self.student(fake, return_features=True)

                loss_KD = - kdloss(s_logit, t_logit)
                loss_G += loss_KD

                if self.dataset == 'cifar10':
                    loss_l2 = - torch.log(F.mse_loss(s_logit, t_logit.detach()))
                        # or loss_l2 = - torch.log(torch.norm(t_logit - s_logit, 2))
                else:
                    loss_l2 = - F.mse_loss(s_logit, t_logit.detach())
                        # or loss_l2 = - torch.log(torch.norm(t_logit - s_logit, 2))
                loss_cadv1 = self.co_alpha * loss_l2
                loss_G += loss_cadv1

                loss_BN = sum([h.r_feature for h in self.hooks])
                loss_G += self.co_beta * loss_BN

                features = torch.cat([s_fea.unsqueeze(1), t_fea.unsqueeze(1)], dim=1)
                sim_criterion = get_sim_criterion(self, self.device)
                loss_ccl = sim_criterion(features, labels=labels)
                loss_G += self.co_gamma * loss_ccl

                loss_ccls = F.cross_entropy(F.softmax(t_logit, dim=1), labels, reduction='mean')

                loss_G += self.co_eta * loss_ccls
            '''
            with torch.no_grad():
                if best_cost > loss_G.item() or best_inputs is None:
                    best_cost = loss_G.item()
                      
                    return {"synthetic": best_inputs}
            '''
            best_inputs = inputs.data
            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

        '''
        '''
        # save best inputs and reset data iter
        self.data_pool.add( best_inputs )
        self.student.train()
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)

    def sample(self):
        return self.data_iter.next()