import torch
import torch.distributed as dist
from models.mtlface import MTLFace

from dataset.preprocess import preprocess

if __name__ == '__main__':

    preprocess('UTK')

    # parser = MTLFace.parser()
    # opt = parser.parse_args()
    # print(opt)

    # dist.init_process_group(backend='gloo', init_method='env://')
    # torch.cuda.set_device(dist.get_rank())
    # model = MTLFace(opt)
    # model.fit()
    # evaluation_accuracy = model.evaluate()
    # with open('Evaluation_Output.txt', 'w') as f:
    #     f.write(str(evaluation_accuracy))

