import torch
import einops as E
import torch.nn.functional as F
import numpy as np
from PIL import Image
from function import *
import os

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dino",
        model_name="vitb16",
        output="dense",
        layer=-1,
        return_multilayer=False
    ):
        super().__init__()
        feat_dims = {
            'vits14': 384,
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }
        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        print(self.checkpoint_name)
        try:
            print(f'Trying loading DINO model from local .cache: ', os.path.expanduser('~')+'/.cache/torch/hub/facebookresearch_dinov2_main')
            dino_vit = torch.hub.load(os.path.expanduser('~')+'/.cache/torch/hub/facebookresearch_dinov2_main', self.checkpoint_name, trust_repo=True, source='local')
        except:
            print(f'Loading model from local fail, try online', f"facebookresearch/{dino_name}", self.checkpoint_name)
            dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        print('finish loading dino')

        self.vit = dino_vit.eval().to(torch.float32)
        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        with torch.no_grad():
            # pad images (if needed) to ensure it matches patch_size
            images = center_padding(images, self.patch_size)
            h, w = images.shape[-2:]
            h, w = h // self.patch_size, w // self.patch_size

            if self.model_name == "dinov2":
                x = self.vit.prepare_tokens_with_masks(images, None)
            else:
                x = self.vit.prepare_tokens(images)

            embeds = []
            for i, blk in enumerate(self.vit.blocks):
                x = blk(x)
                if i in self.multilayers:
                    embeds.append(x)
                    if len(embeds) == len(self.multilayers):
                        break

            num_spatial = h * w
            outputs = []
            for i, x_i in enumerate(embeds):
                cls_tok = x_i[:, 0]
                # ignoring register tokens
                spatial = x_i[:, -1 * num_spatial :]
                x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
                outputs.append(x_i)

            return outputs[0] if len(outputs) == 1 else outputs

if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    import os, pdb, sys
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file_path)
    sys.path.append('Grounded-Segment-Anything')
    from Detect_and_Seg import *
    # detect_and_seg = GroundedSAM(sam_checkpoint=os.path.join(parent_dir, './weights_for_g3flow/sam_vit_h_4b8939.pth'))

    image_path = 'none'
    image = Image.open(image_path)
    images = np.array(image)
    text_prompt = []

    device = 'cuda'
    model = DINO(dino_name="dinov2", model_name='vits14').to(device)
    H, W, _ = images.shape
    feature_map, feature_line = get_dino_feature(images, transform_size=420, model=model)
    res_rgb = feature_to_rgb(feature_line[:, :3], H, W)
    res_rgb = np.array(res_rgb*255, dtype=np.uint8)
    res_rgb = Image.fromarray(res_rgb)
    res_rgb.save('./dino_1.png')

    # =======
    feature_map_tmp = feature_map.permute(2, 0, 1) 
    feature_map_tmp = feature_map_tmp.unsqueeze(0)
    res = PCA_visualize(feature_map_tmp, H, W, return_res=True)
    res = np.array(res*255, dtype=np.uint8)
    res = Image.fromarray(res)
    res.save('./dino_2.png')

    
