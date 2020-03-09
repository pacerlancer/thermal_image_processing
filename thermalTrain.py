import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  
from models import *
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try: 
    from apex import amp
except:
    mixed_precision = False  

wdir = 'weights' + os.sep  
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'


hyp = {'giou': 3.31, 
       'cls': 42.4,  
       'cls_pw': 1.0, 
       'obj': 52.0,  
       'obj_pw': 1.0, 
       'iou_t': 0.213, 
       'lr0': 0.00261,  
       'lrf': -4.,  
       'momentum': 0.949, 
       'weight_decay': 0.000489, 
       'fl_gamma': 0.5, 
       'hsv_h': 0.0103,  
       'hsv_s': 0.691, 
       'hsv_v': 0.433,  
       'degrees': 1.43,  
       'translate': 0.0663, 
       'scale': 0.11,  
       'shear': 0.384}  


f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v


def train():
    cfg = opt.cfg
    data = opt.data
    img_size = opt.img_size
    epochs = 1 if opt.prebias else opt.epochs  
    batch_size = opt.batch_size
    accumulate = opt.accumulate  
    weights = opt.weights  

    if 'pw' not in opt.arc:  
        hyp['cls_pw'] = 1.
        hyp['obj_pw'] = 1.

    
    init_seeds()
    if opt.multi_scale:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  

    
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    
    model = Darknet(cfg, arc=opt.arc).to(device)

    
    pg0, pg1 = [], []  
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  
        else:
            pg0 += [v]  

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  
    del pg0, pg1

    
    cutoff = -1 
    start_epoch = 0
    best_fitness = float('inf')
    attempt_download(weights)
    if weights.endswith('.pt'): 
        
        chkpt = torch.load(weights, map_location=device)

        
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
            
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

       
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0: 
        cutoff = load_darknet_weights(model, weights)

    if opt.transfer or opt.prebias:  
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  

        if opt.prebias:
            for p in optimizer.param_groups:
               
                p['lr'] *= 100 
                if p.get('momentum') is not None:  
                    p['momentum'] *= 0.9

        for p in model.parameters():
            if opt.prebias and p.numel() == nf:  
                p.requires_grad = True
            elif opt.transfer and p.shape[0] == nf:  
                p.requires_grad = True
            else:  
                p.requires_grad = False

    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl', 
                                init_method='tcp://127.0.0.1:9999',  
                                world_size=1, 
                                rank=0)  
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  

    
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,  
                                  rect=opt.rect, 
                                  image_weights=opt.img_weights,
                                  cache_labels=True if epochs > 10 else False,
                                  cache_images=False if opt.prebias else opt.cache_images)

    
    batch_size = min(batch_size, len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16]),
                                             shuffle=not opt.rect, 
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    
    model.nc = nc  
    model.arc = opt.arc  
    model.hyp = hyp  
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  
    torch_utils.model_info(model, report='summary')  
    maps = np.zeros(nc) 
    results = (0, 0, 0, 0, 0, 0, 0)  
    t0 = time.time()
    print('Starting %s for %g epochs...' % ('prebias' if opt.prebias else 'training', epochs))
    for epoch in range(start_epoch, epochs):  
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  
                    p.requires_grad = False if epoch == 0 else True

       
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  
        mloss = torch.zeros(4).to(device)  
        for i, (imgs, targets, paths, _) in pbar: 
            ni = i + nb * epoch  
            imgs = imgs.to(device)
            targets = targets.to(device)

            
            if opt.multi_scale:
                if ni / accumulate % 10 == 0:  
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

           
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

           
            pred = model(imgs)

           
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            
            loss *= batch_size / 64

           
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

           
            mloss = (mloss * i + loss_items) / (i + 1)  
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            
        scheduler.step()

        
        final_epoch = epoch + 1 == epochs
        if opt.prebias:
            print_model_biases(model)
        else:
           
            if not (opt.notest or (opt.nosave and epoch < 10)) or final_epoch:
                with torch.no_grad():
                    results, maps = test.test(cfg,
                                              data,
                                              batch_size=batch_size,
                                              img_size=opt.img_size,
                                              model=model,
                                              conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  
                                              save_json=final_epoch and epoch > 0 and 'coco.data' in data)

        
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n') 
        if len(opt.name) and opt.bucket and not opt.prebias:
            os.system('gsutil cp results.txt gs://%s/results%s.txt' % (opt.bucket, opt.name))

        
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        
        fitness = sum(results[4:])  
        if fitness < best_fitness:
            best_fitness = fitness

        
        save = (not opt.nosave) or (final_epoch and not opt.evolve) or opt.prebias
        if save:
            with open(results_file, 'r') as f:
                
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            
            torch.save(chkpt, last)

           
            if best_fitness == fitness:
                torch.save(chkpt, best)

            )
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

            
            del chkpt

       
    if len(opt.name) and not opt.prebias:
        fresults, flast, fbest = 'results%s.txt' % opt.name, 'last%s.pt' % opt.name, 'best%s.pt' % opt.name
        os.rename('results.txt', fresults)
        os.rename(wdir + 'last.pt', wdir + flast) if os.path.exists(wdir + 'last.pt') else None
        os.rename(wdir + 'best.pt', wdir + fbest) if os.path.exists(wdir + 'best.pt') else None

        
        if opt.bucket:
            os.system('gsutil cp %s %s gs://%s' % (fresults, wdir + flast, opt.bucket))

    plot_results()  
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


def prebias():
   
    if opt.prebias:
        a = opt.img_weights 
        opt.img_weights = False 

        train()  
        create_backbone(last) 

        opt.weights = wdir + 'backbone.pt'  
        opt.prebias = False  
        opt.img_weights = a  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273)  
    parser.add_argument('--batch-size', type=int, default=16) 
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/ultralytics49.pt', help='initial weights')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture') 
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    
    hyp['obj'] *= opt.img_size / 416.

    tb_writer = None
    if not opt.evolve:  
        try:
           
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
        except:
            pass

        prebias()  
        train()  

    else:  
        opt.notest = True  
        opt.nosave = True 
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket) 

        for _ in range(1):  
            if os.path.exists('evolve.txt'): 
               
                x = np.loadtxt('evolve.txt', ndmin=2)
                parent = 'weighted' 
                if parent == 'single' or len(x) == 1:
                    x = x[fitness(x).argmax()]
                    n = min(10, x.shape[0])  
                    x = x[np.argsort(-fitness(x))][:n] 
                    w = fitness(x) - fitness(x).min() 
                    x = (x[:n] * w.reshape(n, 1)).sum(0) / w.sum()  
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 7]

               
                np.random.seed(int(time.time()))
                s = [.2, .2, .2, .2, .2, .2, .2, .0, .02, .2, .2, .2, .2, .2, .2, .2, .2, .2]  
                for i, k in enumerate(hyp.keys()):
                    x = (np.random.randn(1) * s[i] + 1) ** 2.0 

            
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            
            prebias()
            results = train()

           
            print_mutation(hyp, results, opt.bucket)