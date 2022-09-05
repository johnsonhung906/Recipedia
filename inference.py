import os
import multiprocessing
from im2recipe.src.dataset import get_loader
from im2recipe.src.models_new import get_image_model, get_model
import torch.backends.cudnn as cudnn
from im2recipe.src.config import get_args
from tqdm import tqdm
import torch
import numpy as np
import pickle
from PIL import Image
from im2recipe.src.utils.utils import load_checkpoint, count_parameters
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
from transformers import BertTokenizer, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# map_loc = 'cpu'
    

def find_top_k(database, queries, ids, max_k):
    metric='cosine'
    dists = pairwise_distances(queries, database, metric=metric)
    rankings = np.argpartition(dists, range(max_k), axis=-1)[:, :max_k]
    return ids[rankings][0]

def infer(im_path, sim_ingrs):
    args = get_args()
    args.model_name = 'model_pre_best' 
    args.save_dir = './im2recipe/checkpoints' 
    args.embeddings_file = f'./im2recipe/checkpoints/{args.model_name}/feats_test.pkl'
    args.data_dir = './im2recipe/data'
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
    args, pretrained_dict, _ = load_checkpoint(checkpoints_dir, 'best', map_loc, store_dict)
    for var in vars_to_replace:
        setattr(args, var, store_dict[var])

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

    """ simplify ingredients """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    bert = BertModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer(sim_ingrs, return_tensors="pt")
    outputs = bert(**inputs)
    last_hidden_states = outputs.last_hidden_state

    bert_out = last_hidden_states[0,0,:].view(-1, 768)


    """ END OF processing data """
    # layer = 'image_encoder.fc.bias'
    # model = get_image_model(args.output_size, args.backbone)
    model = get_model(args, vocab_size)
    model.load_state_dict(pretrained_dict, strict=False)
    # print(model.state_dict()[layer])
    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()

    # print(model.state_dict()[layer])
    img = img.to(device)
    with torch.no_grad():
        out = model.img_forward(img, bert_out, freeze_backbone=True)

    embedding = out.cpu().detach().numpy()
    print(embedding)
    # print(model.state_dict()[layer])
    """ find corresponding embeddings """
    with open(embeddings_file, 'rb') as f:
        imfeats = pickle.load(f)
        recipefeats = pickle.load(f)
        ids = pickle.load(f)
        ids = np.array(ids)
    
    # find top 3 recipes
    topk = find_top_k(imfeats, embedding, ids, 10)
    # print(topk)
    output = []
    for _id in topk:
        dat = test_data[_id]
        # print(dat)
        title = dat['title']
        instruction = dat['instructions']
        image = dat['images'][0]
        # print(image)
        output.append((title, instruction, image))
        # print(title)

    # print(model.state_dict()[layer])
    return model.named_parameters(), pretrained_dict


im_path = './inversecooking/data/demo_imgs/1.jpg'
sim = 'pepper butter clove oil salt pasta parsley'
outs1, param1 = infer(im_path, sim)