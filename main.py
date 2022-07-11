import os
from datetime import datetime
import argparse
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from glob import glob


from mixModel.__init__ import *
from mixModel.evaluate import evaluate

ex = Experiment('train_transcriber',save_git_info=False)



@ex.config
def config():
    logdir = 'runs/first' # 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 1
    iterations = 20
    resume_iteration = None
    checkpoint_interval = 2000
    train_on = 'Sight to Sound'
    local_rank = -1

    batch_size = 1      #8
    sequence_length = 327680 // 8
    model_complexity = 48

    # if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
    #     batch_size //= 2
    #     sequence_length //= 2
    #     print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 2000

    cross_validation = 5 # K折交叉验证

    ex.observers.append(FileStorageObserver.create(logdir))


def get_kfold_data(X, k, i=0):  
     
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）
    
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid = X[val_start:val_end]
        X_train = X[0:val_start] + X[val_end:]
    else:  # 若是最后一折交叉验证
        X_valid = X[val_start:]     # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        
    return X_train, X_valid


@ex.main
def train(local_rank,logdir, device,epochs, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, cross_validation):
    print_config(ex.current_run)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device=torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    # load data

    data_path = glob(os.path.join('mixModel/data/SIGHT', 'video', 'video_*.mp4'))
    
    train_path, validation_path = data_path[0:-2], data_path[-2:] #get_kfold_data(data_path, cross_validation)
    train_set = SIGHT(sequence_length=sequence_length, groups=['train'], data_path=train_path)
    train_sampler = DistributedSampler(train_set)
    loader = DataLoader(train_set,sampler=train_sampler, batch_size=batch_size, drop_last=True, pin_memory=False)
    validation_dataset = SIGHT(sequence_length=sequence_length, groups=['validation'],data_path=validation_path)
    loader_eval = DataLoader(validation_dataset, 1, shuffle=True, drop_last=True)
    
    # create network and optimizer
    if resume_iteration is None:
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))
    
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    
    
    model.to(device)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
    # loop = tqdm(range(resume_iteration + 1, iterations + 1))   
    # for i, batch in zip(loop, cycle(loader)):

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for i, batch in tqdm(enumerate(loader)):
            if i < resume_iteration:
                continue
            predictions, losses = model.module.run_on_batch(batch)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            if clip_gradient_norm:
                clip_grad_norm_(model.module.parameters(), clip_gradient_norm)
                
            for key, value in {'loss': loss, **losses}.items():
                writer.add_scalar(key, value.item(), global_step=i)
            torch.cuda.empty_cache()

        model.module.eval()
        with torch.no_grad():
            for key, value in evaluate(loader_eval, model.module,save_path=os.path.join(logdir)).items():
                writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
        model.module.train()
        
        torch.save(model.module, os.path.join(logdir, f'model-{resume_iteration+1}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
        
        
if __name__== '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    ex.run(config_updates={'local_rank': args.local_rank})