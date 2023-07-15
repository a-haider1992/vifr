import torch
import torch.distributed as dist
from models.mtlface import MTLFace
import pdb
from dataset.preprocess import preprocess

if __name__ == '__main__':

    # preprocess('UTK')

    parser = MTLFace.parser()
    opt = parser.parse_args()
    print(opt)
    #pdb.set_trace()
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(dist.get_rank())
    model = MTLFace(opt)
    if not opt.evaluation_only:
        model.fit()
    else:
        model.evaluate_mtlface()
