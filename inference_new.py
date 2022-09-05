import os
import multiprocessing
from src.dataset import get_loader
from src.models_new import get_model
import torch.backends.cudnn as cudnn
from src.config import get_args
from tqdm import tqdm
import torch
import numpy as np
import pickle
from PIL import Image
from src.utils.utils import load_checkpoint, count_parameters
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# map_loc = None if torch.cuda.is_available() else 'cpu'
device = 'cpu'
map_loc = 'cpu'
    

def find_top_k(database, queries, ids, max_k):
    metric='cosine'
    dists = pairwise_distances(queries, database, metric=metric)
    rankings = np.argpartition(dists, range(max_k), axis=-1)[:, :max_k]
    return ids[rankings][0]

def infer(args, im_path, sim_ingrs):
    print(f"using device: {device}")
    if device != 'cpu':
        cudnn.benchmark = True
    checkpoints_dir = os.path.join(args.save_dir, args.model_name)
    embeddings_file = args.embeddings_file
    vocab = pickle.load(open(os.path.join(args.data_dir, 'vocab.pkl'), 'rb'))
    vocab_size = len(vocab)
    # print(vocab_size)

    test_data = pickle.load(open(os.path.join(args.data_dir, 'test.pkl'), 'rb'))

    # make sure these arguments are kept from commandline and not from loaded args
    vars_to_replace = ['batch_size', 'eval_split', 'imsize', 'root', 'save_dir']
    store_dict = {}
    for var in vars_to_replace:
        store_dict[var] = getattr(args, var)
    args, pretrained_dict, _ = load_checkpoint(checkpoints_dir, 'best', map_loc,
                                          store_dict)
    for var in vars_to_replace:
        setattr(args, var, store_dict[var])
    
    # print(args)
    """ Processing data """
    transforms_list = [transforms.Resize((args.resize))]

    transforms_list.append(transforms.CenterCrop(args.imsize))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))

    transforms_ = transforms.Compose(transforms_list)

    image = Image.open(im_path)
    img = transforms_(image)
    img = torch.unsqueeze(img, 0)

    title = ""
    ingrs = ""
    instrs = ""

    """ END OF processing data """

    # model = get_model(args.output_size, args.backbone)

    model_dict = model.state_dict()
    model.load_state_dict(pretrained_dict)

    # if device != 'cpu' and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()

    img = img.to(device)
    title = title.to(device)
    ingrs = ingrs.to(device)
    instrs = instrs.to(device)
    sim_ingrs = sim_ingrs.to(device)

    # with torch.no_grad():
    #     out = model(img, title, ingrs, instrs)

    # embedding = out.cpu().detach().numpy()

    # # Load embeddings
    # with open(embeddings_file, 'rb') as f:
    #     imfeats = pickle.load(f)
    #     recipefeats = pickle.load(f)
    #     ids = pickle.load(f)
    #     ids = np.array(ids)
    
    # # find top 3 recipes
    # topk = find_top_k(imfeats, embedding, ids, 5)
    # # print(topk)

    # for _id in topk:
    #     dat = test_data[_id]
    #     title = dat['title']
    #     instructions = dat['instructions']
    #     print(title)


if __name__ == "__main__":
    im_path = '../inversecooking/data/demo_imgs/6.jpg'
    sim_ingrs = ' '.join(['salt', 'vinegar'])
    args = get_args()
    infer(args, im_path, sim_ingrs)
