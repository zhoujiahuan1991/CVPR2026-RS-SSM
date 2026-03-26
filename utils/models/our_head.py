import warnings
import os
from einops import rearrange

from torch import nn
import torch
import torch.nn.functional as F

from .decode_head import BaseDecodeHead_clips_flow_ours

from mmcv.cnn import ConvModule
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from timm.models.layers import DropPath

from .tv3s_head import MLP, Model_type, InferenceParams, Val_type, create_block


from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn


class Interactor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_detail, hidden_semantic):
        concat_feat = torch.cat([hidden_detail, hidden_semantic], dim=-1)
        fused = self.fuse(concat_feat)

        hidden_detail = hidden_detail + self.norm(fused) * self.alpha
        hidden_semantic = hidden_semantic + self.norm(fused) * self.beta

        return hidden_detail, hidden_semantic


class DualPathMambaBlocks(nn.Module):

    def __init__(   self,
                    embed_dim=256,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    depth=24,
                    fused_add_norm=True,
                    residual_in_fp32=True,
                    bimamba=False,
                    mamba2=False,
                    drop_path_rate=0.1,
                    device=None,
                    dtype=None,
                    val_mode=0,
                    model_mode=Model_type.Normal,
                    quad_scan=False):
        
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32

        # Shifted mamba parameters
        self.val_mode = val_mode
        self.model_mode = model_mode

        if model_mode==Model_type.Alt_Residual:
            print("*"*20, "Alternate Residual Connection", "*"*20)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers_detail = nn.ModuleList(
            [
                create_block(
                    d_model=embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    mamba2=mamba2,
                    dual_mamba=True,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.layers_semantic = nn.ModuleList(
            [
                create_block(
                    d_model=embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    mamba2=mamba2,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f_detail = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.norm_f_semantic = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        self.interactors = nn.ModuleList(
            [
                Interactor(dim=embed_dim) for _ in range(depth - 1)
            ]
        )

        self.is_quad_scan = quad_scan


    def flatten(self, x):
        x = rearrange(x, 'b t c h w ->  b (t h w) c')
        return x
    
    def unflatten(self, x, t, h, w):
        x = rearrange(x, 'b (t h w) c -> b t c h w', t=t, h=h, w=w)
        return x

    def forward_feats(self, layer_detail, layer_semantic, hidden_state_detail, hidden_state_semantic, T, H, W,
                    residual_detail=None, residual_semantic=None, inference_params=None):
        # Implementation of inference
        if residual_detail is None:
            res_detail = torch.zeros_like(hidden_state_detail)

        if residual_semantic is None:
            res_semantic = torch.zeros_like(hidden_state_semantic)

        if inference_params is not None:
            if self.val_mode==Val_type.Frame_1_greedy_seq or self.val_mode==Val_type.Frame_4_seq:
                raise NotImplementedError("Sequential Greedy not implemented yet")
            elif self.val_mode == Val_type.Frame_1_greedy:
                raise NotImplementedError("Frame 1 greedy not implemented yet")
            else:
                hidden_state_detail, residual_detail, align_loss = layer_detail(
                    hidden_state_detail, residual_detail, inference_params=inference_params[0], T=T, H=H, W=W
                )
                hidden_state_semantic, residual_semantic = layer_semantic(
                    hidden_state_semantic, residual_semantic, inference_params=inference_params[1]
                )
        else:   # Train and val for non sequential frame-by-frame processing
            hidden_state_detail, residual_detail, align_loss = layer_detail(
                hidden_state_detail, residual_detail, inference_params=None, T=T, H=H, W=W
            )
            hidden_state_semantic, residual_semantic = layer_semantic(
                hidden_state_semantic, residual_semantic, inference_params=None
            )

        if residual_detail is None:
            residual_detail = res_detail

        if residual_semantic is None:
            residual_semantic = res_semantic

        return hidden_state_detail, hidden_state_semantic, residual_detail, residual_semantic, align_loss


    def forward(self, x_detail, x_semantic, inference_params=None):
        # mamba impl
        residual_detail = None
        residual_semantic = None
        hidden_state_detail = x_detail
        hidden_state_semantic = x_semantic

        B, T, Cd, Hd, Wd = hidden_state_detail.shape
        _, _, Cs, Hs, Ws = hidden_state_semantic.shape

        hidden_state_detail = self.flatten(hidden_state_detail)
        hidden_state_semantic = self.flatten(hidden_state_semantic)
        if residual_detail is not None:
            residual_detail = self.flatten(residual_detail)
        if residual_semantic is not None:
            residual_semantic = self.flatten(residual_semantic)

        align_loss = 0.0
        for idx in range(len(self.layers_detail)):
            layer_detail = self.layers_detail[idx]
            layer_semantic = self.layers_semantic[idx]

            if idx % 2 == 0:
                if self.is_quad_scan:
                    raise NotImplementedError("Quad scan not implemented yet")
                    #hidden_state_detail = torch.rot90(hidden_state_detail, 1, (-1, -2))
                    #hidden_state_semantic = torch.rot90(hidden_state_semantic, 1, (-1, -2))

            A_log = layer_semantic.mixer.A_log
            A_log = A_log.detach()
            layer_detail.mixer.set_A_log(A_log)

            in_proj_weight = layer_semantic.mixer.in_proj.weight
            in_proj_weight = in_proj_weight.detach()
            in_proj_bias = layer_semantic.mixer.in_proj.bias
            if in_proj_bias is not None:
                in_proj_bias = in_proj_bias.detach()
            layer_detail.mixer.set_in_proj_weight(in_proj_weight, in_proj_bias)

            #x_proj_weight = layer_semantic.mixer.x_proj.weight
            #x_proj_weight = x_proj_weight.detach()
            #layer_detail.mixer.set_x_proj_weight(x_proj_weight)

            hidden_state_detail, hidden_state_semantic, residual_detail, residual_semantic, layer_loss = self.forward_feats(
                layer_detail,
                layer_semantic,
                hidden_state_detail,
                hidden_state_semantic,
                T, Hd, Wd,
                residual_detail,
                residual_semantic,
                inference_params
            )
            align_loss += layer_loss

            if idx < len(self.layers_detail) - 1:
                hidden_state_detail, hidden_state_semantic = self.interactors[idx](
                    hidden_state_detail, hidden_state_semantic
                )

        if not self.fused_add_norm:
            if residual_detail is None:
                residual_detail = hidden_state_detail
            else:
                residual_detail = residual_detail + self.drop_path(hidden_state_detail)
            hidden_state_detail = self.norm_f_detail(residual_detail.to(dtype=self.norm_f_detail.weight.dtype))

            if residual_semantic is None:
                residual_semantic = hidden_state_semantic
            else:
                residual_semantic = residual_semantic + self.drop_path(hidden_state_semantic)
            hidden_state_semantic = self.norm_f_semantic(residual_semantic.to(dtype=self.norm_f_semantic.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f_detail, RMSNorm) else layer_norm_fn
            hidden_state_detail = fused_add_norm_fn(
                self.drop_path(hidden_state_detail),
                self.norm_f_detail.weight,
                self.norm_f_detail.bias,
                eps=self.norm_f_detail.eps,
                residual=residual_detail,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f_semantic, RMSNorm) else layer_norm_fn
            hidden_state_semantic = fused_add_norm_fn(
                self.drop_path(hidden_state_semantic),
                self.norm_f_semantic.weight,
                self.norm_f_semantic.bias,
                eps=self.norm_f_semantic.eps,
                residual=residual_semantic,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if residual_detail is not None:
            residual_detail = self.unflatten(residual_detail, T, Hd, Wd)
        if residual_semantic is not None:
            residual_semantic = self.unflatten(residual_semantic, T, Hs, Ws)
        # hidden_states = rearrange(hidden_states, '(b i j) (t h w) c -> b t c (i h) (j w)', i=h1,    j=w1, h=self.window_h1 , w=self.window_w1)  
        hidden_state_detail = self.unflatten(hidden_state_detail, T, Hd, Wd)
        hidden_state_semantic = self.unflatten(hidden_state_semantic, T, Hs, Ws)

        if inference_params is not None:
            self.residual_detail = residual_detail
            self.residual_semantic = residual_semantic

        return hidden_state_detail, hidden_state_semantic, align_loss


@HEADS.register_module()
class Ours_MambaHead(BaseDecodeHead_clips_flow_ours):
    '''
    Note: Updated with mamba blocks from videomamba
    '''
    def __init__(self, feature_strides, **kwargs):
        super(Ours_MambaHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        num_frames = kwargs['num_clips']
        embedding_dim = decoder_params['embed_dim']
        self.emmbedding_dim = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.greedy_inference = False
        self.inference_params = None
        self.test_mode = False
        self.val_mode = Val_type.Normal
        self.use_emmbedding = False
        self.prev_class = None
        self.add_mode = False
        self.concat_mode = False

        if 'quad_scan' in decoder_params:
            self.quad_scan = decoder_params['quad_scan']
        else:
            self.quad_scan = False
        
        if 'model_type' in decoder_params:
            self.model_type = decoder_params['model_type']
        else:
            self.model_type = Model_type.Normal

        if not 'n_mambas' in decoder_params:
            total_mambas = 8                   # Note the default is 8 for shifted mambas
        else:
            total_mambas = decoder_params['n_mambas']

        if not 'mamba2' in decoder_params:
            mamba2 = False
        else:
            mamba2 = decoder_params['mamba2']

        if 'test_mode' in decoder_params:
            self.test_mode = decoder_params['test_mode']
            if self.test_mode:
                if 'max_seqlen' in decoder_params and 'max_batch_size' in decoder_params:
                    NotImplementedError("Inference params not implemented for user params")
                else:
                    print("Inference params not provided, using default of 1000*1000 for seqlen and 10 for batch size")
                    self.max_seqlen = 1000*1000
                    self.max_batch_size = 10
                    self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size)]

                if 'val_mode' in decoder_params:
                    self.val_mode = decoder_params['val_mode']

                    if self.val_mode == Val_type.Frame_1_greedy or self.val_mode == Val_type.Frame_1_greedy_seq:
                        self.greedy_inference = True
                        print("*"*20, "Running Greedy Infer!!", "*"*20)
                    elif self.val_mode == Val_type.Frame_4_seq:
                        # self.greedy_inference = True
                        print("*"*20, "Running Seq Infer with F4!!", "*"*20)
                else:
                    self.val_mode = Val_type.Normal

        if self.model_type == Model_type.Add:
            self.add_mode = True
        elif self.model_type == Model_type.Concat:
            self.concat_mode = True

        if self.model_type == Model_type.Bi or self.model_type == Model_type.Bi_Embed:
            bimamba = True
        else:
            bimamba = False

        if bimamba:
            if 'im_size' in decoder_params:
                self.h, self.w = decoder_params['im_size']
                if 'num_clips' in decoder_params:
                    num_frames = decoder_params['num_clips']
                else:
                    warnings.warn("n_frames not provided, using default of 512")
                    num_frames = 512
            else:
                warnings.warn("im_size not provided, using default of 60x60")
                self.h, self.w = 60, 60
                warnings.warn("n_frames not provided, using default of 512")
                num_frames = 512

            if self.model_type == Model_type.Bi_Embed:
                self.use_emmbedding = True
                self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, self.emmbedding_dim, 1, 1, 1, 1))
                self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.emmbedding_dim, (self.h//self.window_h), 1, (self.w//self.window_w), 1))

            # If testing
            if self.test_mode:
                self.t_count = 0

        # self.vs = Mamba(d_model=embedding_dim, d_state=16, d_conv=4, expand=2)
        # self.vs = mambaSequential()
        self.vs = DualPathMambaBlocks(
            embed_dim=embedding_dim,
            depth=total_mambas,
            bimamba=bimamba,
            mamba2=mamba2,
            val_mode=self.val_mode,
            model_mode=self.model_type,
            quad_scan=self.quad_scan
        )

        self.edge_fusion = ConvModule(
            in_channels=embedding_dim*2,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.linear_predvs = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        if self.concat_mode:
            self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)
        else:
            self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.linear_pred_detail = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.detail_scale = nn.Parameter(torch.tensor(0.1))


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temporal_pos_embedding', 'pos_embed'}
    
    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None, img_metas=None):
        if self.training:
            assert self.num_clips==num_clips, f"{self.num_clips} and {num_clips}"

        # Assert for img_metas is not None during inference
        if self.test_mode and self.inference_params is not None:
            assert img_metas is not None, "Require img_metas during inference to properly clear the statespace"
            assert len(img_metas) == 1, "Only single image inference supported" 

            # Support only single GPU inference
            cur_class = os.path.split(os.path.split(os.path.split(img_metas[0]['filename'])[0])[0])[1]           
            if self.prev_class is not None and self.prev_class != cur_class:
                self.inference_params = [InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size), InferenceParams(max_seqlen=self.max_seqlen,max_batch_size=self.max_batch_size)]

                '''Changed here as was not previously reset'''
                # self.inference_params.seqlen_offset = 0 
                self.inference_params[0].seqlen_offset = 0 
                self.vs.residual = None

                if type(self.inference_params[1]) is list:
                    for i in range(len(self.inference_params[1])):
                        self.inference_params[1][i].seqlen_offset = 0
                else:
                    self.inference_params[1].seqlen_offset = 0

                if self.use_emmbedding:
                    self.t_count = 0

                print(cur_class)

            self.prev_class = cur_class

        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32 # No Chg
        # [[8, 96, 120, 120], [8, 192, 60, 60], [8, 384, 30, 30], [8, 768, 15, 15]]
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # [8, 768, 15, 15] -> [8, 256, 15, 15]
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 15, 15] -> [8, 256, 120, 120]

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]) # [8, 384, 30, 30] -> [8, 256, 30, 30]
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 30, 30] -> [8, 256, 120, 120]

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]) # [8, 192, 60, 60] -> [8, 256, 60, 60]
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False) # [8, 256, 60, 60] -> [8, 256, 120, 120]

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]) # [8, 96, 120, 120] -> [8, 256, 120, 120]

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # [8, 256, 120, 120] 

        _, _, h, w=_c.shape

        if not self.test_mode:
            x = self.dropout(_c) # [8, 256, 120, 120]
            x = self.linear_pred(x) # [8, 256, 120, 120] -> [8, 124, 120, 120]
            x = x.reshape(batch_size, num_clips, -1, h, w) # [8, 124, 120, 120] -> [2, 4, 124, 120, 120]
        else:
            x = self.linear_pred(_c) # [8, 256, 120, 120] -> [8, 124, 120, 120]
            x = x.reshape(batch_size, num_clips, -1, h, w) # [8, 124, 120, 120] -> [2, 4, 124, 120, 120]
            # x = torch.zeros(batch_size, num_clips, self.num_classes, h, w).to(_c.device)

        # print("_c.shape: ", _c.shape)
        # print("Checking....")
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]
        # if n ==1: 
        #     return x[:,-1]
        # print("Not Skipped!!!!")

        h2, w2 = int(h/2), int(w/2)
        #h2, w2 = int(h * 3 / 4), int(w * 3 / 4)

        #detail_feat = self.fuser(c1, c2, c3, c4)  # H/4×W/4×C_detail , H/8×W/8×C_semantic ↓
        _c = resize(_c, size=(h2,w2),mode='bilinear', align_corners=False) # [8, 256, 120, 120] -> [8, 256, 60, 60]

        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2) # [8, 256, 60, 60] -> [2, 4, 256, 60, 60]
        semantic_feat = _c_further
        detail_feat = _c_further.clone()

        #detail_feat = rearrange(detail_feat, '(b t) c h w -> b t c h w', b=batch_size, t=num_clips)       # [B, T, Cd, Hd, Wd]
        #semantic_feat = rearrange(semantic_feat, '(b t) c h w -> b t c h w', b=batch_size, t=num_clips)   # [B, T, Cs, Hs, Ws]

        # Note the shape change
 
        if self.inference_params is not None and not self.greedy_inference:
            self.inference_params[0].seqlen_offset = 0 
            self.vs.residual = None

            if type(self.inference_params[1]) is list:
                for i in range(len(self.inference_params[1])):
                    self.inference_params[1][i].seqlen_offset = 0
            else:
                self.inference_params[1].seqlen_offset = 0
                
            detail_feat, semantic_feat, align_loss = self.vs(detail_feat, semantic_feat, self.inference_params)
        else:
            detail_feat, semantic_feat, align_loss = self.vs(detail_feat, semantic_feat, self.inference_params)


        if self.concat_mode:
            raise NotImplementedError # [2, 256, 60, 60] + [2, 256, 60, 60] -> [2, 512, 60, 60]
        elif self.add_mode:
            raise NotImplementedError # [2, 256, 60, 60] + [2, 256, 60, 60] -> [2, 256, 60, 60]
        else:
            _c_further_detail = detail_feat[:, -1] # B Cd Hd Wd
            _c_further_semantic = semantic_feat[:, -1] # B Cs Hs Ws

            #Hs, Ws = _c_further_semantic.shape[2], _c_further_semantic.shape[3]
            #_c_further_detail = resize(_c_further_detail, size=(Hs, Ws), mode='bilinear', align_corners=False) * self.detail_scale # [2, 64, 60, 60]
            _c_further2 = torch.cat([_c_further_semantic, _c_further_detail * self.detail_scale], dim=1)  # [2, 256+256, 60, 60]
            _c_further2 = self.edge_fusion(_c_further2)  # [2, 256, 60, 60]

        x2 = self.dropout(_c_further_semantic) # [2, 256, 60, 60]
        x2 = self.linear_pred2(x2) # [2, 256, 60, 60] -> [2, 124, 60, 60]
        x2 = resize(x2, size=(h,w),mode='bilinear',align_corners=False) # [2, 124, 60, 60] -> [2, 124, 120, 120]


        detail_pred = self.dropout(_c_further_detail) # [2, 256, 60, 60]
        detail_pred = self.linear_pred_detail(detail_pred) # [2, 256, 60, 60] -> [2, 124, 60, 60]
        detail_pred = resize(detail_pred, size=(h,w),mode='bilinear',align_corners=False) # [2, 124, 120, 120]


        x2 = x2.unsqueeze(1) # [2, 124, 120, 120] -> [2, 1, 124, 120, 120]

        x = torch.cat([x,x2],1)   ## b*(k+1)*124*h*w [2, 5, 124, 120, 120]

        if not self.training:
            return x2.squeeze(1)

        return x, detail_pred, align_loss
        #return x
