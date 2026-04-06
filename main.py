# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
from copyreg import pickle
import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, build_class_weights, save_checkpoint, get_emotion_labels, compute_war_uar,save_csv_report, save_confusion_matrix
import pickle
from train import train_epoch
from validation import val_epoch
import time


EARLY_STOPPING_PATIENCE = 30   # epochs without composite improvement before stopping
MIN_EPOCH_FOR_BEST       = 10  # fixed warm-up buffer — no unfreeze epoch to anchor to


if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 12
    test_accuracies = []
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    pretrained = opt.pretrain_path != 'None'    
    
    #opt.result_path = 'res_'+str(time.time())
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    
    opt.arch = '{}'.format(opt.model)  
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
        

    annotation_path = opt.annotation_path
    emotion_labels = get_emotion_labels(opt.dataset)
    base_dir = opt.result_path

    
    for fold in range(n_folds):
        #if opt.dataset == 'RAVDESS':
        #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'


        fisher_path = None
        if opt.fisherindex_template is not None:
            fisher_path = opt.fisherindex_template.format(fold=fold + 1)
            if not os.path.exists(fisher_path):
                print(f"[Fisher] WARNING: indices file not found at "
                        f"{fisher_path} — using all channels")
                fisher_path = None


        train_loss_history = {}
        train_acc_history  = {}
        train_uar_history = {}
        val_loss_history   = {}
        val_acc_history    = {}
        val_uar_history = {}

        fold_dir = os.path.join(base_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)

        early_stop = False


        if opt.dataset == 'RAVDESS':
           opt.annotation_path = (annotation_path +
                                       f'/annotations_croppad_fold{fold+1}.txt')
        
        print(opt)
        with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)
            
        torch.manual_seed(opt.manual_seed)

        _tmp_audio_channels = opt.audio_channels
        if fisher_path is not None:
            _tmp_audio_channels = len(np.load(fisher_path))
        model, parameters = generate_model(
            opt, audio_input_channels=_tmp_audio_channels)
        criterion = nn.CrossEntropyLoss().to(opt.device)
        
        if not opt.no_train:
            
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])
            
            # audio_transform = transforms.Compose([
            #     transforms.NormalizeAudio(),
            #     transforms.RandomGainVariation(min_gain=0.7, max_gain=1.3),  # 2. gain shift
            #     transforms.RandomNoiseInjection(min_snr_db=20, max_snr_db=40),  # 3. noise
            #     transforms.RandomTimeShift(opt.max_shift_ratio)],
            #     )

            training_data = get_training_set(opt, spatial_transform=video_transform, fisher_indices_path=fisher_path) 
            train_class_weights = build_class_weights(
                    training_data, opt.n_classes, opt.device)
            criterion_train = nn.CrossEntropyLoss(weight=train_class_weights)
            criterion_train = criterion_train.to(opt.device)
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
        
            # train_logger = Logger(
            #     os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
            #     ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            # train_batch_logger = Logger(
            #     os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
            #     ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
            

            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
            
        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])     
            # audio_transform = transforms.Compose([
            #     transforms.NormalizeAudio()])
            validation_data = get_validation_set(opt, spatial_transform=video_transform, fisher_indices_path=fisher_path)
            

            val_class_weights = build_class_weights(
                    validation_data, opt.n_classes, opt.device)
            criterion_val = nn.CrossEntropyLoss(weight=val_class_weights)
            criterion_val = criterion_val.to(opt.device)
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            
            patience_counter = 0
            best_loss      = 1e10
            best_uar       = 0
            best_composite = 0.0
            # val_logger = Logger(
            #         os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            # test_logger = Logger(
            #         os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            
        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            if early_stop:
                    print(f"[Early Stopping] Triggered at epoch {i} — "
                          f"no composite improvement for "
                          f"{EARLY_STOPPING_PATIENCE} epochs.")
                    break
            
            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_loss, train_acc, y_true, y_pred = train_epoch(i, train_loader, model, criterion, optimizer, opt)
                war, uar = compute_war_uar(y_true, y_pred)
                train_loss_history[i] = float(train_loss)
                train_acc_history[i]  = float(train_acc)
                train_uar_history[i] = float(uar)
                save_checkpoint(
                        {
                            'epoch':      i,
                            'arch':       opt.arch,
                            'state_dict': model.state_dict(),
                            'optimizer':  optimizer.state_dict(),
                            'best_prec1': best_prec1,
                            'best_composite':   best_composite,
                        },
                        False, opt, fold, fold_dir)
            
            if not opt.no_val:
                val_loss, prec1, y_true, y_pred = val_epoch(i, val_loader, model, criterion, opt)
                war, uar = compute_war_uar(y_true, y_pred)
                composite = 0.7 * uar + 0.3 * war



                val_loss_history[i] = float(val_loss)
                val_acc_history[i]  = float(prec1)
                val_uar_history[i] = float(uar)
                print(f"  Epoch {i:3d}  val_acc={prec1:.4f}"
                        f"  WAR={war:.4f}  UAR={uar:.4f}"
                        f"  composite={composite:.4f}"
                        f"  patience={patience_counter}/{EARLY_STOPPING_PATIENCE}")

                is_best = (composite > best_composite) and (i >= MIN_EPOCH_FOR_BEST)

                if i >= MIN_EPOCH_FOR_BEST:
                    if composite > best_composite:
                        best_composite = composite
                        best_prec1     = prec1
                        patience_counter = 0
                        print(f"  [Early Stopping] Composite improved to "
                                f"{best_composite:.4f} — counter reset.")
                    else:
                        patience_counter += 1
                        if patience_counter >= EARLY_STOPPING_PATIENCE:
                            early_stop = True

                save_checkpoint(
                    {
                        'epoch':          i,
                        'arch':           opt.arch,
                        'state_dict':     model.state_dict(),
                        'optimizer':      optimizer.state_dict(),
                        'best_prec1':     best_prec1,
                        'best_composite': best_composite,
                    },
                    is_best, opt, fold, fold_dir)
            if not opt.no_train:
                # scheduler.step(val_loss)
                for name, hist in [
                    (f'train_loss_{fold+1}.pkl', train_loss_history),
                    (f'train_acc_{fold+1}.pkl', train_acc_history),
                    (f'train_uar_{fold+1}.pkl',  train_uar_history),
                    (f'val_loss_{fold+1}.pkl',   val_loss_history),
                    (f'val_acc_{fold+1}.pkl',    val_acc_history),
                    (f'val_uar_{fold+1}.pkl',  val_uar_history),
                ]:
                    with open(os.path.join(fold_dir, name), 'wb') as f:
                        pickle.dump(hist, f)
        if early_stop:
            print(f"[Fold {fold+1}] Stopped early at epoch {i}. "
                      f"Best composite: {best_composite:.4f}")         
        if opt.test:

            # test_logger = Logger(
            #         os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
            # audio_transform = transforms.Compose([
            #     transforms.NormalizeAudio()])
            test_data = get_test_set(opt, spatial_transform=video_transform,  fisher_indices_path=fisher_path) 
            test_class_weights = build_class_weights(
                    validation_data, opt.n_classes, opt.device)
            criterion_test = nn.CrossEntropyLoss(weight=test_class_weights)
            criterion_test = criterion_test.to(opt.device)
            path = '%s/%s_best' % (fold_dir, opt.store_name)+str(fold)+'.pth'
            #load best model
            best_state = torch.load(path,map_location=opt.device)
            model.load_state_dict(best_state['state_dict'])
        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            test_loss, test_prec1, y_true, y_pred = val_epoch(
                    10000, test_loader, model, criterion, opt)

            war, uar = compute_war_uar(y_true, y_pred)

            save_csv_report(y_true, y_pred, fold_dir, fold, emotion_labels)
            save_confusion_matrix(y_true, y_pred, fold_dir, fold,
                                      emotion_labels)
            print(f"Fold {fold+1}  Acc={test_prec1:.4f}"
                      f"  WAR={war:.4f}  UAR={uar:.4f}")
            with open(os.path.join(fold_dir,
                        f'test_summary_fold_{fold+1}.txt'), 'w') as f:
                f.write(f'Dataset: {opt.dataset}\n')
                f.write(f'Fold: {fold+1}\n')
                f.write(f'Precision@1: {test_prec1:.4f}\n')
                f.write(f'Loss: {test_loss:.4f}\n')
                f.write(f'WAR: {war:.4f}\n')
                f.write(f'UAR: {uar:.4f}\n')

            test_accuracies.append(
                test_prec1.cpu().item()
                if isinstance(test_prec1, torch.Tensor)
                else float(test_prec1)
            )
        if opt.test and len(test_accuracies) == n_folds:
            acc_array = np.array([
                acc.cpu().item() if isinstance(acc, torch.Tensor) else acc
                for acc in test_accuracies
            ])
            print(f"\n{'='*70}")
            print(f'FINAL RESULTS - {opt.dataset}')
            print(f"{'='*70}")
            print(f'Mean Accuracy: '
                  f'{np.mean(acc_array):.4f} ± '
                  f'{np.std(acc_array):.4f}')
            print(f'All fold accuracies: {acc_array}')

            summary_file = os.path.join(opt.result_path, 'test_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f'Dataset: {opt.dataset}\n')
                f.write(f'Model: {opt.model}\n')
                f.write(f'Number of folds: {n_folds}\n')
                f.write(f'Mean Accuracy: '
                        f'{np.mean(acc_array):.4f} ± '
                        f'{np.std(acc_array):.4f}\n')
                f.write('Individual fold accuracies:\n')
                for idx, acc in enumerate(acc_array):
                    f.write(f'  Fold {idx+1}: {acc:.4f}\n')
            print(f'\nSummary saved to: {summary_file}')