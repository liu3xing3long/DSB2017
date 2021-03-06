from preprocessing import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas

datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']
bbox_result_path = './bbox_result'
feature_path = config_submit['feature_path']

if not skip_prep:
    testsplit = full_prep(datapath,prep_result_path,
                          n_worker = config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
else:
    testsplit = os.listdir(datapath)

if not skip_detect:
    nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    checkpoint = torch.load(config_submit['detector_param'])
    nod_net.load_state_dict(checkpoint['state_dict'])

    torch.cuda.set_device(0)
    nod_net = nod_net.cuda()
    cudnn.benchmark = True
    nod_net = DataParallel(nod_net)

    if not os.path.exists(bbox_result_path):
        os.mkdir(bbox_result_path)
    #testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)

    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])

casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
casenet = casemodel.CaseNet(topk=5)
config2 = casemodel.config
checkpoint = torch.load(config_submit['classifier_param'])
casenet.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
casenet = casenet.cuda()
cudnn.benchmark = True
casenet = DataParallel(casenet)

filename = config_submit['outputfile']



def test_casenet(model,testset):
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 2,
        pin_memory=False)
    #model = model.cuda()
    model.eval()
    predlist = []

    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):
        ################################
        data_name = testset.filenames[i]
        idxtoken1 = data_name.rindex("/")
        idxtoken2 = data_name.rindex("_")
        data_name = data_name[idxtoken1 + 1:idxtoken2]
        print "processing {}".format(data_name)
        ################################
        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        nodulePred, casePred, _, noduleFeat, centerFeat = model(x,coord)
        # print "data {}, nod {}, case {}".format(i, nodulePred, casePred)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
        # feat_shape = centerFeat.size()
        # centerFeat = centerFeat.view(feat_shape[0] * feat_shape[1], -1)

        ################################
        np.save(os.path.join(feature_path, data_name + "_feat.npy"), centerFeat.data.cpu().numpy())

    predlist = np.concatenate(predlist)
    return predlist    
config2['bboxpath'] = bbox_result_path
config2['datadir'] = prep_result_path



dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
predlist = test_casenet(casenet,dataset).T
df = pandas.DataFrame(data={'id':testsplit, 'cancer':predlist})
df.to_csv(filename,index=False)
