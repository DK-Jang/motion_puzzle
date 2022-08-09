import os
import sys
import copy
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('./etc')
from utils import get_model_list
logger = logging.getLogger(__name__)

from model import (Generator)
from radam import RAdam


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.gen = Generator(config['model']['gen'])
        self.gen_ema = copy.deepcopy(self.gen)

        self.model_dir = config['model_dir']
        self.config = config

        lr_gen = config['lr_gen']
        gen_params = list(self.gen.parameters())
        self.gen_opt = RAdam([p for p in gen_params if p.requires_grad],
                              lr=lr_gen, weight_decay=config['weight_decay'])

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.gen = nn.DataParallel(self.gen).to(self.device)
            self.gen_ema = nn.DataParallel(self.gen_ema).to(self.device)

    def train(self, loader, wirter):
        config = self.config

        def run_epoch(epoch):
            self.gen.train()
            
            pbar = tqdm(enumerate(zip(loader['train_src'], loader['train_tar'])), 
                        total=len(loader['train_src']))
            
            for it, (con_data, sty_data) in pbar:
                gen_loss_total, gen_loss_dict = self.compute_gen_loss(con_data, sty_data)
                self.gen_opt.zero_grad()
                gen_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
                self.gen_opt.step()
                update_average(self.gen_ema, self.gen)

                # report progress
                log = "Epoch [%i/%i], " % (epoch+1, config['max_epochs'])
                all_losses = dict()
                for loss in [gen_loss_dict]:
                    for key, value in loss.items():
                        if key.find('total') > -1:
                            all_losses[key] = value
                log += ' '.join(['%s: [%.2f]' % (key, value) for key, value in all_losses.items()])
                pbar.set_description(log)

                if (it+1) % config['log_every'] == 0:
                    for k, v in gen_loss_dict.items():
                        wirter.add_scalar(k, v, epoch*len(loader['train_src'])+it)
                        
        for epoch in range(config['max_epochs']):
            run_epoch(epoch)

            if (epoch+1) % config['save_every'] == 0:
                self.save_checkpoint(epoch+1)
    
    def compute_gen_loss(self, xa_data, xb_data):
        config = self.config

        xa = xa_data['motion'].to(self.device)
        xb = xb_data['motion'].to(self.device)

        xaa, xbb, xab, xaba, xabb = self.gen(xa, xb)

        loss_recon = F.l1_loss(xaa, xa) + F.l1_loss(xbb, xb)
        loss_cyc_con = F.l1_loss(xaba, xa)
        loss_cyc_sty = F.l1_loss(xabb, xb)
        loss_sm_rec = F.l1_loss((xaa[..., :-1] - xaa[..., 1:]), (xa[..., :-1] - xa[..., 1:])) + \
                       F.l1_loss((xbb[..., :-1] - xbb[..., 1:]), (xb[..., :-1] - xb[..., 1:]))
        loss_sm_cyc = F.l1_loss((xaba[..., :-1] - xaba[..., 1:]), (xa[..., :-1] - xa[..., 1:])) + \
                       F.l1_loss((xabb[..., :-1] - xabb[..., 1:]), (xb[..., :-1] - xb[..., 1:]))

        # summary
        l_total = (config['rec_w'] * loss_recon
                 + config['cyc_con_w'] * loss_cyc_con
                 + config['cyc_sty_w'] * loss_cyc_sty
                 + config['sm_rec_w'] * loss_sm_rec
                 + config['sm_cyc_w'] * loss_sm_cyc)
            
        l_dict = {'loss_total': l_total,
                  'loss_recon': loss_recon,
                  'loss_cyc_con': loss_cyc_con,
                  'loss_cyc_sty': loss_cyc_sty,
                  'loss_sm_rec': loss_sm_rec,
                  'loss_sm_cyc': loss_sm_cyc}

        return l_total, l_dict
    
    @torch.no_grad()
    def test(self, xa, xb):
        config = self.config
        self.gen_ema.eval()

        xaa, xbb, xab, xaba, xabb = self.gen_ema(xa, xb, phase='test')

        loss_recon = F.l1_loss(xaa, xa) + F.l1_loss(xbb, xb)
        loss_cyc_con = F.l1_loss(xaba, xa)
        loss_cyc_sty = F.l1_loss(xabb, xb)
        loss_sm_rec = F.l1_loss((xaa[..., :-1] - xaa[..., 1:]), (xa[..., :-1] - xa[..., 1:])) + \
                       F.l1_loss((xbb[..., :-1] - xbb[..., 1:]), (xb[..., :-1] - xb[..., 1:]))
        loss_sm_cyc = F.l1_loss((xaba[..., :-1] - xaba[..., 1:]), (xa[..., :-1] - xa[..., 1:])) + \
                       F.l1_loss((xabb[..., :-1] - xabb[..., 1:]), (xb[..., :-1] - xb[..., 1:]))

        # summary
        l_total = (config['rec_w'] * loss_recon
                 + config['cyc_con_w'] * loss_cyc_con
                 + config['cyc_sty_w'] * loss_cyc_sty
                 + config['sm_rec_w'] * loss_sm_rec
                 + config['sm_cyc_w'] * loss_sm_cyc)
            
        l_dict = {'loss_total': l_total,
                  'loss_recon': loss_recon,
                  'loss_cyc_con': loss_cyc_con,
                  'loss_cyc_sty': loss_cyc_sty,
                  'loss_sm_rec': loss_sm_rec,
                  'loss_sm_cyc': loss_sm_cyc}
        
        out_dict = {
            "recon_con": xaa,
            "stylized": xab,
            "con_gt": xa,
            "sty_gt": xb
        }

        return out_dict, l_dict

    def save_checkpoint(self, epoch):
        gen_path = os.path.join(self.model_dir, 'gen_%03d.pt' % epoch)

        # DataParallel wrappers keep raw model object in .module attribute
        raw_gen = self.gen.module if hasattr(self.gen, "module") else self.gen
        raw_gen_ema = self.gen_ema.module if hasattr(self.gen_ema, "module") else self.gen_ema

        logger.info("saving %s", gen_path)
        torch.save({'gen': raw_gen.state_dict(), 
                    'gen_ema': raw_gen_ema.state_dict()}, gen_path)
        
        print('Saved model at epoch %d' % epoch)
    
    def load_checkpoint(self, model_path=None):
        if not model_path:
            model_dir = self.model_dir
            model_path = get_model_list(model_dir, "gen")   # last model

        state_dict = torch.load(model_path, map_location=self.device)
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_ema.load_state_dict(state_dict['gen_ema'])

        epochs = int(model_path[-6:-3])
        print('Load from epoch %d' % epochs)

        return epochs
    
    # def load_checkpoint(self, model_path=None):
    #     if not model_path:
    #         model_dir = self.model_dir
    #         model_path = get_model_list(model_dir, "gen")   # last model

    #     map_location = lambda storage, loc: storage
    #     if torch.cuda.is_available():
    #         map_location = None
    #     state_dict = torch.load(model_path, map_location=map_location)

    #     # if self.device == 'cpu':
    #     #     state_dict = torch.load(model_path, map_location=self.device)
    #     # else: 
    #     #     state_dict = torch.load(model_path)

    #     self.gen.load_state_dict(state_dict['gen'])
    #     self.gen_ema.load_state_dict(state_dict['gen_ema'])

    #     epochs = int(model_path[-6:-3])
    #     print('Load from epoch %d' % epochs)

    #     return epochs

def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


if __name__ == '__main__':
    import argparse
    from etc.utils import get_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the config file.')
    args = parser.parse_args()
    config = get_config(args.config)
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")

    trainer = Trainer(config)

    xa = torch.randn(1, 12, 21, 240)
    xb = torch.randn(1, 12, 21, 120)
    xa_foot = torch.zeros(1, 240, 4)

    xa_data = {'motion': xa}
    xb_data = {'motion': xb}

    trainer.compute_gen_loss(xa_data, xb_data)

    # print(in_xb1)

    