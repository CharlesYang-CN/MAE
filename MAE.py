import torch
import torch.nn as nn
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, indexes.unsqueeze(-1).expand(-1, -1, sequences.shape[-1]))

class PatchShuffle(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [np.random.permutation(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack(indexes, axis=-1), dtype=torch.long).to(patches.device)
        unmasked_indexes = forward_indexes[:remain_T]  # 未被掩码 patch 的原始索引

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, unmasked_indexes
    
class MAE_Encoder(nn.Module):
    def __init__(self, 
                 image_size=224,
                 patch_size=16,
                 emb_dim=384,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75
                ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1,1,emb_dim),requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.zeros((image_size//patch_size)**2,1,emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self,img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes
    
class MAE_Decoder(nn.Module):
    def __init__(self, image_size=224,
                 patch_size=16,
                 emb_dim=384,
                 num_layer=4,
                 num_head=3,
                 ):
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, unmasked_indexes):
        T, B, C = features.shape  # T = remain_T + 1 (包括 cls_token)
        T_total = (224 // 16) ** 2 + 1  # 总 patch 数（包括 cls_token）

        # 创建全为掩码 token 的特征张量
        full_features = self.mask_token.expand(T_total, B, -1)

        # 放置 cls_token
        full_features[0] = features[0]

        # 将未被掩码的 patch 放回原始位置
        full_features[unmasked_indexes + 1] = features[1:]  # +1 因为 cls_token 占第 0 位

        full_features = full_features + self.pos_embedding

        full_features = rearrange(full_features, 't b c -> b t c')
        full_features = self.transformer(full_features)
        full_features = rearrange(full_features, 'b t c -> t b c')

        patches = self.head(full_features[1:])  # 移除 cls_token
        mask = torch.ones_like(patches)  # 被掩码位置为 1
        mask[unmasked_indexes] = 0  # 未被掩码位置为 0
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask
    
class MAE_ViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=384,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask