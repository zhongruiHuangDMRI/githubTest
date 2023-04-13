import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import lib.utils as utils


class TensorboardWriter_v2():

    def __init__(self, args):
        name_model = args.log_dir + args.model + "_" + args.dataset_name + "_" + utils.datestr()
        self.writer = SummaryWriter(log_dir=args.log_dir + name_model, comment=name_model)
        utils.make_dirs(args.save)
        self.csv_train, self.csv_val, self.csv_test = self.create_stats_files(args.save)
        self.dataset_name = args.dataset_name
        self.classes = args.classes
        self.label_names = 'dMRI'
        self.data={"train":{},"val":{},"test":{}}
        self.data['train']['loss'] = 0.0
        self.data['val']['loss'] = 0.0
        self.data['test']['loss'] = 0.0
        self.data['train']['count'] = 0.0
        self.data['val']['count'] = 0.0
        self.data['test']['count'] = 0.0

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, 'train.csv'), 'w')
        val_f = open(os.path.join(path, 'val.csv'), 'w')
        test_f = open(os.path.join(path, 'test.csv'), 'w')
        return train_f, val_f,test_f

    def update_scores(self, mode, loss, epoch, epoch_lenth, batch_idx):
        self.data[mode]['loss'] += loss
        self.data[mode]['count'] = batch_idx+1
        self.display_tensorboard(mode, loss, epoch*epoch_lenth+batch_idx)
        self.display_terminal(mode, loss, epoch, batch_idx)
    
    def display_tensorboard(self, mode, loss, global_step):
        if self.writer is not None:
            self.writer.add_scalar(mode,loss,global_step)
    
    def display_terminal(self, mode, loss, epoch, batch_idx):
        info_print = "\n Mode:{} Epoch:{} Batch_idx: {} Loss:{:.4f} ".format(mode, epoch, batch_idx, self.data[mode]['loss'] / self.data[mode]['count'])                              
        print(info_print)

    def reset(self, mode):
        self.data[mode]['loss'] = 0.0
        self.data[mode]['count'] = 1

    def write_end_of_epoch(self, type, epoch):
        if type=="train":
            train_csv_line = 'Epoch:{:2d} Loss:{:.4f}'.format(epoch,self.data['train']['loss'] / self.data['train']['count'])                                            
            val_csv_line = 'Epoch:{:2d} Loss:{:.4f}'.format(epoch,self.data['val']['loss'] / self.data['val']['count'])
            self.csv_train.write(train_csv_line + '\n')
            self.csv_val.write(val_csv_line + '\n')
        if type=="test":
            test_csv_line = 'Epoch:{:2d} Loss:{:.4f}'.format(epoch,self.data['test']['loss'] / self.data['test']['count'])
            self.csv_test.write(test_csv_line + '\n')