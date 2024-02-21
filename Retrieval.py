import argparse
import os
import numpy as np
import time
import datetime
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, model_without_ddp, train_loader, val_loader, test_loader, train_args, tokenizer):
    
    max_epoch = config.schedular['epochs']
    batch_size_train = config.batch_size_train
    K = config.gradient_accumulation_steps
    logging_steps = config.logging_steps
    logging_strategy = config.logging_strategy
    ckpt_output_path = config.ckpt_output_path
    max_grad_norm = config.max_grad_norm
    metrics = config.metrics
    optimizer, lr_scheduler = train_args['optimizer'], train_args['lr_scheduler']
    start_epoch = train_args.get('start_epoch', 1)
    total_step = train_args.get('total_step', 0)
    total_train_batch_size = batch_size_train * K * config.world_size

    best_scores = defaultdict(lambda:None)
    scaler = GradScaler()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    logger.info("Start training")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num Epochs = {max_epoch}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_train}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {K}")
    start_time = time.time()    
    
    for epoch in range(start_epoch, max_epoch+1):
        model.train()
        logger.info(" -" * 20 + "Epochs of [{}/{}]".format(epoch, max_epoch) + " - " * 20)
        header = 'Train Epoch: [{}/{}]'.format(epoch, max_epoch)
        if config.distributed:
            train_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
    
        for i, (image, text, idx) in enumerate(metric_logger.log_every(train_loader, print_freq=500, header=header)):
            image = image.to(device,non_blocking=True)   
            idx = idx.to(device,non_blocking=True)   
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)  

            alpha = config['alpha'] 
            if epoch == 0 and config['warm_up']:
                alpha *= min(1, i/len(train_loader))

            sync_context = model.no_sync if config.local_rank != -1 and (i + 1) % K != 0 else nullcontext
            amp_context = autocast if config.use_amp else nullcontext
            with sync_context():
                with amp_context():
                    loss_ita, loss_itm = model(image, text_input,alpha=alpha, idx=idx)                  
                    loss = loss_ita + loss_itm          
                    loss = loss / K
                scaler.scale(loss).backward()

            if (i + 1) % K == 0:
                # Best practice of AMP in DDP framework:
                # https://pytorch.org/docs/stable/notes/amp_examples.html#functions-that-need-a-particular-dtype
                # https://zhuanlan.zhihu.com/p/165152789
                scaler.unscale_(optimizer)                
                # https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
                # https://blog.csdn.net/zhaohongfei_358/article/details/122820992
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
            
            total_step += 1 
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss_ita=loss_ita.item())
            metric_logger.update(loss_itm=loss_itm.item())
            metric_logger.synchronize_between_processes()
            need_tb_logs = metric_logger.latest_meter(prefix='train/')
            
            if (logging_strategy == "epoch" and i == len(train_loader) - 1) or (logging_strategy == "steps" and total_step % logging_steps == 0):
                val_stats, val_prediction = evaluate(model_without_ddp, val_loader, tokenizer)
                test_stats, test_prediction = evaluate(model_without_ddp, test_loader, tokenizer, "Test") if not config.only_dev else (val_stats, val_prediction)

                save_evidence = []
                for metric_name in metrics:
                    if metric_name not in val_stats.keys():
                        continue
                    score = best_scores[metric_name]
                    current_score = float(val_stats[metric_name])
                    if score is None or current_score > score:
                        save_evidence.append(metric_name)
                        best_scores[metric_name] = current_score

                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'step': i+1,
                             'total_step': total_step
                             }

                need_tb_logs.update({
                    **{f'val/{k}': v for k, v in val_stats.items()},
                    **{f'test/{k}': v for k, v in test_stats.items()}
                })

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    'step': i+1,
                    'total_step': total_step
                }

                if utils.is_main_process():
                    ckpt_sub_path = os.path.join(ckpt_output_path, f"epoch_{epoch}-step_{i}")

                    # logging statements
                    utils.write_json(ckpt_sub_path, "log_stats", log_stats)

                    # logging prediction
                    utils.write_json(ckpt_sub_path, "val_prediction", val_prediction)
                    utils.write_json(ckpt_sub_path, "test_prediction", test_prediction)

                    # Saving normal checkpoints
                    if save_evidence or config.save_every_checkpoint:
                        torch.save(save_obj, os.path.join(ckpt_sub_path, 'checkpoint.pth'))

                    # Saving checkpoints if they are distinct
                    for metric_name in save_evidence:
                        best_ckpt_path = os.path.join(ckpt_output_path, f"best_{metric_name}")
                        utils.copy_whole_dir(ckpt_sub_path, best_ckpt_path)

            tb_writer.add_dict_scalar(need_tb_logs, total_step)
            lr_scheduler.step(total_step)

        if utils.is_dist_avail_and_initialized():
            dist.barrier()
        torch.cuda.empty_cache()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: {}".format(metric_logger.summary(mode="avg")))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('***** Stopping training *****')
    logger.info('Training time {}'.format(total_time_str))
    tb_writer.close() 



@torch.no_grad()
def evaluate(model, data_loader, tokenizer, special_name="Val"):
    logger.info("- - - - - - - - - - - - - Evaluation- - - - - - - - - - - - - ")
    start_time = time.time()  
    # test
    model.eval()
    metric_logger = utils.MetricLogger(logging=logger.info, delimiter=" - ")
    header = f'Evaluating {special_name} Set: '

    texts = data_loader.dataset.text
    origin_texts=data_loader.dataset.origin_text
    txt2img = data_loader.dataset.txt2img
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    
    for i in metric_logger.log_every(range(0, num_text, text_bs), header=header + "Loading text features..."):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed.cpu())   
        text_feats.append(text_feat.cpu())
        text_atts.append(text_input.attention_mask.cpu())
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    
    image_feats = []
    image_embeds = []
    image_ids = []
    img2txt = data_loader.dataset.img2txt
    for image, img_id in metric_logger.log_every(data_loader, header=header + "Loading image features..."): 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed.cpu())
        image_ids.extend(img_id.tolist())
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header + "image2text...")): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[topk_idx].to(device), 
                                    attention_mask = text_atts[topk_idx].to(device),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header + "text2image...")): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1).to(device), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1).to(device),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    if config.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Evaluation time {}'.format(total_time_str)) 

    logger.info("- - - - - - - - - - - - - Calculating Results- - - - - - - - - - - - - ")
    result = {}
    topk_upper = config['k_test']
    scores_i2t = score_matrix_i2t.cpu().numpy()
    scores_t2i = score_matrix_t2i.cpu().numpy()
    # Get topk retrieval results
    for img in range(scores_i2t.shape[0]):
        contents = []
        for wait_check in np.argsort(scores_i2t[img])[::-1][:topk_upper]:
            contents.append(int(wait_check))
        result[image_ids[img]] = contents

    topk_result = {}
    for image_id, txt_ids in result.items():
        topk_result[data_loader.dataset.image[image_id]] = {
            "goldens": [origin_texts[txtid] for txtid in img2txt[image_id]], 
            "topks": [origin_texts[txtid] for txtid in txt_ids]
            }
    # Images->Text
    pres = np.zeros((scores_i2t.shape[0], 10))
    ranks = np.zeros(scores_i2t.shape[0])
    golden_total=0
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        goldens = img2txt[index]
        golden_total+=len(goldens)
        for i in goldens:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        pres[index] = np.cumsum(np.in1d(inds[:10], goldens))

    # Compute metrics

    pr5 = 100.0 * np.sum(pres[:, 4]) / golden_total
    pr10 = 100.0 * np.sum(pres[:, 9]) / golden_total

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        copes = np.where(inds == txt2img[index])[0]
        if len(copes) == 0:
            ranks[index] = 0
        else:
            ranks[index] = copes[0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    pr_mean = (pr5 + pr10)/2
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'i2t_r1': tr1,
                   'i2t_r5': tr5,
                   'i2t_r10': tr10,
                   'i2t_r_mean': tr_mean,
                   'i2t_pr5': pr5,
                   'i2t_pr10': pr10,
                   'i2t_pr_mean': pr_mean,
                   't2i_r1': ir1,
                   't2i_r5': ir5,
                   't2i_r10': ir10,
                   't2i_r_mean': ir_mean,
                   'r_mean': r_mean
                   }
    logger.info(eval_result)
    return eval_result, topk_result




def main():

    cudnn.benchmark = True

    #### Dataset ####
    logger.info("- - - - - - - - - - - - - Creating dataset- - - - - - - - - - - - - ")
    train_dataset, val_dataset, test_dataset = create_dataset('re', config)  

    if config.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None, None, None])    
    # next(iter(train_loader))
    tokenizer = BertTokenizer.from_pretrained(config.text_encoder)

    #### Model ####
    train_args = {}
    logger.info("- - - - - - - - - - - - - Creating model- - - - - - - - - - - - - ")
    model = ALBEF(config=config, text_encoder=config.text_encoder, tokenizer=tokenizer)
    
    if config.checkpoint:    
        checkpoint = torch.load(config.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        logger.info('load checkpoint from %s'%config.checkpoint)
        logger.info(msg)  
        
    
    model = model.to(device)   
    
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module   
    
    #### Training Controler ####
    logger.info("- - - - - - - - - - - - - Loading TrainArgs- - - - - - - - - - - - - ")
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche.steps_per_epoch = len(train_loader)
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    train_args['optimizer'] = optimizer
    train_args['lr_scheduler'] = lr_scheduler

    if config.eval_before_train:
        logger.info("- - - - - - - - - - - - - Evaluate Before Train- - - - - - - - - - - - - ")
        evaluate(model_without_ddp, val_loader, tokenizer)
        evaluate(model_without_ddp, test_loader, tokenizer, special_name="Test")

    train(model, model_without_ddp, train_loader, val_loader,
                  test_loader, train_args, tokenizer)     
    logger.info("- - - - - - - - - - - - - End of All- - - - - - - - - - - - - ")            


def parse_args():
    parser = argparse.ArgumentParser(
        description="necessarily parameters for run this code."
    )     
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='../Output/ALBEF/Retrieval_coco')        
    parser.add_argument('--checkpoint', default='../Models/ALBEF/ALBEF_4M.pth')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_backend', default='nccl')
    parser.add_argument('--eval_before_train', action='store_true')
    parser.add_argument('--only_dev', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='gradient accumulation for increase batch virtually.')
    parser.add_argument('--max_grad_norm', default=5.0, type=float,
                        help='clip gradient norm of an iterable of parameters')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='device number of current process.')
    parser.add_argument('--logging_steps', default=500, type=int)
    parser.add_argument('--logging_strategy', type=str, choices=['no','epoch','steps'], default='epoch')
    parser.add_argument('--logging_level', type=str, choices=['DEBUG','INFO','ERROR','WARNING'], default='INFO')
    parser.add_argument('--save_every_checkpoint', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    config = parser.parse_args()
    return config
            
if __name__ == '__main__':

    # set configuration for training or evaluating
    args = parse_args()
    config = utils.read_yaml(args.config)
    config = utils.AttrDict(config)
    args = utils.AttrDict(args.__dict__)
    # The parameters passed in from the command line take precedence
    config.update(args)

    # Determine global parameters and settings
    utils.init_distributed_mode(config)
    device = torch.device(config.device)
    # fix the seed for reproducibility
    utils.setup_seed(config.seed)
    # record them in file.
    current_branch, git_info = utils.get_git_info(os.path.dirname(os.path.abspath(__file__)))
    config.current_branch = current_branch
    logger, tb_writer = utils.create_logger(config)

    logger.info(f"Here is all global configuration:\n {str(config)}")
    logger.info(f"Here is all git repo infomation:\n {git_info}")

    main()
