from PIL import Image
import requests
import pickle
from io import BytesIO
import gradio as gr
from inversecooking.src.args import get_parser
from inversecooking.src.model import get_model  as get_model1
import torch
import os
from inversecooking.src.model1_inf import im2ingr
import numpy as np

response = requests.get("https://img.freepik.com/free-photo/chicken-wings-barbecue-sweetly-sour-sauce-picnic-summer-menu-tasty-food-top-view-flat-lay_2829-6471.jpg?w=2000")
dog_img = Image.open(BytesIO(response.content))

def img2ingr(image):
    if image:
        # print(image)
        # img_file = '../data/demo_imgs/1.jpg'
        img = Image.open(image).convert('RGB')
        # img = Image.fromarray(np.uint8(image)).convert('RGB')
        ingr = im2ingr(img, ingrs_vocab, model)
        return ' '.join(ingr)
    else:
        return ''

def img_ingr2recipe(im_path, ingr):
    def output_recipe(outs):
        return outs[0].upper() + "\n-----------------\n" + "\n".join([f'{i+1}. {step}' for i, step in enumerate(outs[1])]) 
    
    # print(im_path, ingr)
    outs_all = infer(im_path)
    return_recipe1 = output_recipe(outs_all[0])
    return_recipe2 = output_recipe(outs_all[1])
    return_recipe3 = output_recipe(outs_all[2])
    
    return dog_img, return_recipe1, dog_img, return_recipe2, dog_img, return_recipe3

def change_checkbox(predicted_ingr):
    return gr.update(label="Ingredient required", interactive=True, choices=predicted_ingr.split(), value=predicted_ingr.split())

def add_ingr(new_ingr):
    print(new_ingr)
    return "hello"

def add_to_checkbox(old_ingr, new_ingr):
    # chack if in dict or not
    return gr.update(label="Ingredient required", interactive=True, choices=[*old_ingr, new_ingr], value=[*old_ingr, new_ingr])


""" load model1 """
args = get_parser()

# basic parameters
model_dir = './inversecooking/data'
data_dir = './inversecooking/data'
example_dir = './inversecooking/data/demo_imgs/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

# load ingredients vocab
ingrs_vocab = pickle.load(open(os.path.join(model_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)

# model setting and loading
args.maxseqlen = 15
args.ingrs_only=True
model = get_model1(args, ingr_vocab_size, instrs_vocab_size)
model_path = os.path.join(model_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()
model.ingrs_only = True
model.recipe_only = False

""" load model2 """
import os
import multiprocessing
from im2recipe.src.dataset import get_loader
from im2recipe.src.models import get_model as get_model2
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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# map_loc = None if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# map_loc = 'cpu'

def find_top_k(database, queries, ids, max_k):
    metric='cosine'
    dists = pairwise_distances(queries, database, metric=metric)
    rankings = np.argpartition(dists, range(max_k), axis=-1)[:, :max_k]
    return ids[rankings][0]

def infer(im_path):
    args = get_args()
    args.model_name = 'r50_ssl' 
    args.save_dir = './im2recipe/checkpoints' 
    args.embeddings_file = './im2recipe/checkpoints/r50_ssl/feats_test.pkl'
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

    """ END OF processing data """

    model = get_model2(args, vocab_size)

    model.load_state_dict(pretrained_dict, strict=False)
    # print(model.state_dict()[layer])
    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()

    # print(model.state_dict()[layer])
    img = img.to(device)
    with torch.no_grad():
        out = model.img_forward(img, freeze_backbone=True)

    embedding = out.cpu().detach().numpy()
    # print(embedding)
    # print(model.state_dict()[layer])
    # Load embeddings
    with open(embeddings_file, 'rb') as f:
        imfeats = pickle.load(f)
        recipefeats = pickle.load(f)
        ids = pickle.load(f)
        ids = np.array(ids)
    
    # find top 3 recipes
    topk = find_top_k(imfeats, embedding, ids, 3)
    # print(topk)
    titles, instructions = [], []
    output = []
    for _id in topk:
        dat = test_data[_id]
        title = dat['title']
        instruction = dat['instructions']
        output.append((title, instruction))
    
    return output


""" gradio """
# input image -> list all required ingrs -> checkbox for selecting ingrs / input_box for input more ingrs user want -> output: recipe and its image
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Recipedia
    Start finding the yummy recipe ...
    """)
    with gr.Tabs():
        with gr.TabItem("User"):
            # input image
            image_input = gr.Image(label="Upload the image of your yummy food", type='filepath')
            gr.Examples(examples=[example_dir+"1.jpg", example_dir+"2.jpg", example_dir+"3.jpg", example_dir+"4.jpg", example_dir+"5.jpg", example_dir+"6.jpg"], inputs=image_input)
            with gr.Row():
                # clear_img_btn = gr.Button("Clear")
                image_btn = gr.Button("Upload", variant="primary")
            # list all required ingrs -> checkbox for selecting ingrs / input_box for input more ingrs user want
            predicted_ingr = gr.Textbox(visible=False)

            with gr.Row():
                checkboxes = gr.CheckboxGroup(label="Ingredient required", interactive=True)
                new_ingr = gr.Textbox(label="Addtional ingredients", max_lines=1)
                    # with gr.Row():
                    #     new_btn_clear = gr.Button("Clear")
                    #     new_btn = gr.Button("Add", variant="primary")

            add_ingr = gr.Textbox(visible=False)

            with gr.Row():
                clear_ingr_btn = gr.Button("Reset")
                ingr_btn = gr.Button("Confirm", variant="primary")

            # output: recipe and its image
            with gr.Row():
                out_recipe1 = gr.Textbox(label="Your recipe 1", value="Spagetti ---\n1. cook it!")
                out_image1 = gr.Image(label="Looks yummy ><")
            with gr.Row():
                out_recipe2 = gr.Textbox(label="Your recipe 2", value="Spagetti ---\n1. cook it!")
                out_image2 = gr.Image(label="Looks yummy ><")
            with gr.Row():
                out_recipe3 = gr.Textbox(label="Your recipe 3", value="Spagetti ---\n1. cook it!")
                out_image3 = gr.Image(label="Looks yummy ><")

        with gr.TabItem("Example"):
            image_button = gr.Button("Flip")
        
        image_btn.click(img2ingr, inputs=image_input, outputs=predicted_ingr)
        predicted_ingr.change(fn=change_checkbox, inputs=predicted_ingr, outputs=checkboxes)

        # new_btn.click(img2ingr, inputs=new_ingr, outputs=predicted_ingr)
        new_ingr.submit(fn=add_to_checkbox, inputs=[checkboxes, new_ingr], outputs=checkboxes)

        ingr_btn.click(img_ingr2recipe, inputs=[image_input, checkboxes], outputs=[out_image1, out_recipe1, out_image2, out_recipe2, out_image3, out_recipe3])


demo.launch(debug=True, share=True)