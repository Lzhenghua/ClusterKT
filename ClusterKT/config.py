from argparse import ArgumentParser
import torch

def set_opt(dataset):
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use. default to -1: not using any')
    if dataset = 'Junyi':
        arg_parser.add_argument("--learning_rate", dest="learning_rate", default=0.001, type=float, required=False)
        arg_parser.add_argument("--kq_same", dest="kq_same", default=1, type=int, required=False)
        arg_parser.add_argument("--n_blocks", dest="n_blocks", default=4, type=int, required=False)
        arg_parser.add_argument("--cluster_size", dest="memory_size", default=40, type=int, required=False)
        arg_parser.add_argument("--batch_size", dest="batch_size",  default=64, type=int, required=False)
        arg_parser.add_argument("--time", dest="time", default=300, type=int, required=False)
        arg_parser.add_argument("--interval", dest="interval", default=1440, type=int, required=False)
        arg_parser.add_argument("--final_fc_dim", dest="final_fc_dim", default=512, type=int, required=False)
        arg_parser.add_argument("--n_heads", dest="n_heads", default=8, type=int, required=False)
        arg_parser.add_argument("--d_ff", dest="d_ff", default=1024, type=int, required=False)
        arg_parser.add_argument("--embed_dim", dest="embed_dim",default=256, type=int, required=False)
        arg_parser.add_argument("--dropout", dest="dropout", default=0.05, type=float, required=False)
        arg_parser.add_argument("--epoch", dest="epoch", default=100, type=int, required=False)
        arg_parser.add_argument("--max_len", dest="max_len", default=100, type=int, required=False)
        arg_parser.add_argument("--save_params", dest="save_params", default=False, type=bool, required=False)
        
    elif dataset = 'NIPS34':
        arg_parser.add_argument("--learning_rate", dest="learning_rate", default=0.001, type=float, required=False)
        arg_parser.add_argument("--kq_same", dest="kq_same", default=1, type=int, required=False)
        arg_parser.add_argument("--n_blocks", dest="n_blocks", default=4, type=int, required=False)
        arg_parser.add_argument("--cluster_size", dest="memory_size", default=10, type=int, required=False)
        arg_parser.add_argument("--batch_size", dest="batch_size",  default=32, type=int, required=False)
        arg_parser.add_argument("--time", dest="time", default=300, type=int, required=False)
        arg_parser.add_argument("--interval", dest="interval", default=1440, type=int, required=False)
        arg_parser.add_argument("--final_fc_dim", dest="final_fc_dim", default=512, type=int, required=False)
        arg_parser.add_argument("--n_heads", dest="n_heads", default=8, type=int, required=False)
        arg_parser.add_argument("--d_ff", dest="d_ff", default=1024, type=int, required=False)
        arg_parser.add_argument("--embed_dim", dest="embed_dim",default=256, type=int, required=False)
        arg_parser.add_argument("--dropout", dest="dropout", default=0.05, type=float, required=False)
        arg_parser.add_argument("--epoch", dest="epoch", default=100, type=int, required=False)
        arg_parser.add_argument("--max_len", dest="max_len", default=100, type=int, required=False)
        arg_parser.add_argument("--save_params", dest="save_params", default=False, type=bool, required=False)
        
    elif dataset = 'Assist15':
        arg_parser.add_argument("--learning_rate", dest="learning_rate", default=0.001, type=float, required=False)
        arg_parser.add_argument("--kq_same", dest="kq_same", default=1, type=int, required=False)
        arg_parser.add_argument("--n_blocks", dest="n_blocks", default=4, type=int, required=False)
        arg_parser.add_argument("--cluster_size", dest="memory_size", default=10, type=int, required=False)
        arg_parser.add_argument("--batch_size", dest="batch_size",  default=32, type=int, required=False)
        arg_parser.add_argument("--time", dest="time", default=300, type=int, required=False)
        arg_parser.add_argument("--interval", dest="interval", default=1440, type=int, required=False)
        arg_parser.add_argument("--final_fc_dim", dest="final_fc_dim", default=512, type=int, required=False)
        arg_parser.add_argument("--n_heads", dest="n_heads", default=8, type=int, required=False)
        arg_parser.add_argument("--d_ff", dest="d_ff", default=1024, type=int, required=False)
        arg_parser.add_argument("--embed_dim", dest="embed_dim",default=256, type=int, required=False)
        arg_parser.add_argument("--dropout", dest="dropout", default=0.05, type=float, required=False)
        arg_parser.add_argument("--epoch", dest="epoch", default=100, type=int, required=False)
        arg_parser.add_argument("--max_len", dest="max_len", default=100, type=int, required=False)
        arg_parser.add_argument("--save_params", dest="save_params", default=False, type=bool, required=False)
        
    else:
      arg_parser.add_argument("--learning_rate", dest="learning_rate", default=0.001, type=float, required=False)
        arg_parser.add_argument("--kq_same", dest="kq_same", default=1, type=int, required=False)
        arg_parser.add_argument("--n_blocks", dest="n_blocks", default=4, type=int, required=False)
        arg_parser.add_argument("--cluster_size", dest="memory_size", default=10, type=int, required=False)
        arg_parser.add_argument("--batch_size", dest="batch_size",  default=64, type=int, required=False)
        arg_parser.add_argument("--time", dest="time", default=300, type=int, required=False)
        arg_parser.add_argument("--interval", dest="interval", default=1440, type=int, required=False)
        arg_parser.add_argument("--final_fc_dim", dest="final_fc_dim", default=512, type=int, required=False)
        arg_parser.add_argument("--n_heads", dest="n_heads", default=8, type=int, required=False)
        arg_parser.add_argument("--d_ff", dest="d_ff", default=1024, type=int, required=False)
        arg_parser.add_argument("--embed_dim", dest="embed_dim",default=256, type=int, required=False)
        arg_parser.add_argument("--dropout", dest="dropout", default=0.05, type=float, required=False)
        arg_parser.add_argument("--epoch", dest="epoch", default=100, type=int, required=False)
        arg_parser.add_argument("--max_len", dest="max_len", default=100, type=int, required=False)
        arg_parser.add_argument("--save_params", dest="save_params", default=False, type=bool, required=False)
    opt = parser.parse_args()
    if opt.gpu == -1 or torch.cuda.is_available() == False:
        opt.DEVICE = 'cpu'
        opt.gpu = -1
    else:
        opt.DEVICE = opt.gpu
    print(opt)
    return opt
