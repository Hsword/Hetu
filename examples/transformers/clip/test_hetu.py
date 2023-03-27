import argparse
from PIL import Image
import torch
import hetu as ht
import numpy as np
from six.moves import urllib
import os
from hetu_clip import CLIPModel
from config import CLIPConfig
import requests
from hetu.transforms import Compose, Resize, CenterCrop, Normalize


def convert_image_to_rgb(img):
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr /= 255
    return arr
    
def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def tokenize(texts, context_length=77, truncate=False):
    if not os.path.exists('vocab.json'):
        origin = ("https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json")
        print('Downloading vocab from %s' % origin)
        urllib.request.urlretrieve(origin, "vocab.json")
    if not os.path.exists('merges.txt'):
        origin = ("https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt")
        print('Downloading vocab from %s' % origin)
        urllib.request.urlretrieve(origin, "merges.txt")        
    tokenizer = ht.tokenizers.CLIPTokenizer('vocab.json', 'merges.txt')
    
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    result = np.zeros((len(texts), context_length))
    
    for i in range(len(texts)):
        text = texts[i]
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)         
        result[i][0] = sot_token

        if len(ids) + 2 > context_length:
            if truncate:
                result[i][1:context_length-1] = ids[:context_length-2]
                result[i][-1] = eot_token
            else:
                raise RuntimeError(f"Input {text} is too long for context length {context_length}")
        result[i, 1:1+len(ids)] = ids
        result[i, 1+len(ids)] = eot_token
    return result

def build_model(state_dict):

    vision_width = state_dict["vision_model.encoder.layers.0.self_attn.out_proj.weight"].shape[0]
    vision_inter_width = state_dict["vision_model.encoder.layers.0.mlp.fc1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("vision_model.encoder.layers.") and k.endswith(".self_attn.out_proj.weight")])
    vision_patch_size = state_dict["vision_model.embeddings.patch_embedding.weight"].shape[-1]
    grid_size = round((state_dict["vision_model.embeddings.position_embedding.weight"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["visual_projection.weight"].shape[0]
    context_length = state_dict["text_model.embeddings.position_embedding.weight"].shape[0]
    vocab_size = state_dict["text_model.embeddings.token_embedding.weight"].shape[0]
    transformer_width = state_dict["text_model.embeddings.token_embedding.weight"].shape[1]
    transformer_inter_width = state_dict["text_model.encoder.layers.0.mlp.fc1.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len([k for k in state_dict.keys() if k.startswith("text_model.encoder.layers.") and k.endswith(".self_attn.out_proj.weight")])

    image_config = {'hidden_size':vision_width, 
                    'intermediate_size':vision_inter_width,
                    'image_size':image_resolution, 
                    'num_hidden_layers':vision_layers, 
                    'patch_size':vision_patch_size,
                    'hidden_act':'gelu'}
    text_config = {'vocab_size':vocab_size,
                   'hidden_size':transformer_width, 
                   'intermediate_size':transformer_inter_width,
                   'num_hidden_layers':transformer_layers,
                   'num_attention_heads':transformer_heads,
                   'max_position_embeddings':context_length,
                   'hidden_act':'gelu'}
                    
    config = CLIPConfig(text_config, image_config, projection_dim=embed_dim)
    model = CLIPModel(config)
    return model

def test(args):
    device_id=args.gpu_id
    executor_ctx = ht.gpu(device_id)
    
    if not os.path.exists('pytorch_model.bin'):
        origin = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin"
        print('Downloading model from %s' % origin)
        urllib.request.urlretrieve(origin, 'pytorch_model.bin')    
        
    state_dict = torch.load('pytorch_model.bin')
    vision_patch_size = state_dict["vision_model.embeddings.patch_embedding.weight"].shape[-1]
    grid_size = round((state_dict["vision_model.embeddings.position_embedding.weight"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    pixel_values_shape = (1, 3, image_resolution, image_resolution)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image = transform((image_resolution, image_resolution))(image).reshape(1, image_resolution, image_resolution, 3).transpose(0, 3, 1, 2)
    
    text = tokenize(["a photo of a cat", "a photo of a dog"])
    input_ids_shape = text.shape
        
    model = build_model(state_dict)
    
    pixel_values = ht.Variable(name='pixel_values', trainable=False)
    input_ids = ht.Variable(name='input_ids', trainable=False)
 
    result = model(input_ids, input_ids_shape, pixel_values, pixel_values_shape)  
    probs = ht.softmax_op(result[0])
    executor = ht.Executor([probs],ctx=executor_ctx)
    
    model_dict = {key:state_dict[key].cpu().numpy() for key in state_dict}
    for node in executor.param_nodes:
        pre_shape = executor.config.placeholder_to_arr_map[node].shape
        value = model_dict[node.name]
        if node.name=='logit_scale':
            executor.config.placeholder_to_arr_map[node][:] = value
            continue
        cur_shape = value.shape
        assert pre_shape == cur_shape, 'Shape not conform! Got {} and {} for {}.'.format(pre_shape, cur_shape, node.name)
        executor.config.placeholder_to_arr_map[node][:] = value

    feed_dict = {input_ids: text, pixel_values: image}
    results = executor.run(feed_dict = feed_dict)
    prob = results[0].asnumpy()
    print("Label probs:", prob)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--model", type=str, default="ViT-B-16", help="The model to test."
    )
    args = parser.parse_args()
    test(args)
