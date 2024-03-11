# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import sys 
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import DistanceDropEdge
from utils import TemporalData
from utils import init_weights 



# 考虑了时空数据的特性
class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel

        # 按照距离进行Local Region划分 
        # 50m范围划分Local Region  
        # 返回mask后的edge_index & edge_attr 
        self.drop_edge = DistanceDropEdge(local_radius)  
        
        # Agent-to-agent Interaction 学习每个Timestep，每个Local Region中，中心Agent与相邻Agent之间的关系
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel) 
        
        # Temporal Dependency 捕获每个局部区域的时间信息 
        # 每个Agent i，输入是上个模块输入的所有{s_i^T}_{t=1}^T，并加入额外可学习的Token s_{i}^{T+1}
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers) 
        
        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)

    def forward(self, data: TemporalData) -> torch.Tensor: 
        # 在每个tp进行循环
        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
        if self.parallel:
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                snapshots[t] = Data(x=data.x[:, t], edge_index=edge_index, edge_attr=edge_attr,
                                    num_nodes=data.num_nodes)
            batch = Batch.from_data_list(snapshots)
            out = self.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                                  bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            out = [None] * self.historical_steps
            for t in range(self.historical_steps): 
                
                # 1. 划分Local Region 
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}']) 
                
                # 2. 每个tp提取Agent-to-agent的Feature 
                out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr, bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat']) 
                
            # 3. Stack时间维度上所有的AA Feature 
            out = torch.stack(out)  # [T, N, D] 
            
        # 4. Temporal Dependency --- Self-Attention，时间维度上的融合
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps]) 
        
        # 5. 获取Local Region内的道路Node和Edge 
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors']) 
        
        # 6. Agent-to-Lane Encoder 
        out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                              is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
                              traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat']) 
        
        # 7. Output Feature (融合后的Feature)
        return out


# Agent-to-agent Encoder (消息传递)   
# 每个tp，每个local region中，中心Agent与相邻Agent之间的关系 
# Key: 引入Rotation Invariant的交叉注意力机制来聚合空间信息 
class AAEncoder(MessagePassing): 
    """ 
    关于Message Passing 基类 
    ----------------------------------------------
    x_i^k = \gamma^k (x_i^{k-1}, agg_{j∈N(i)}Message^k(x_i^{k-1},x_j^{k-1}, e_j,i))
    
    - x_i^k-1: 节点i在k-1层特征   
    - e_{i,j}: 是节点j->i的边特征（非必须） 
    
    三个关键步骤: 
    - 消息message  
        基于函数Message(..)来定义每个邻居节点传递给中心节点i的消息 
    - 聚合aggregate 
        得到邻居传递给中心节点的消息后，基于聚合函数来聚合邻域消息，这个函数是置换不变的(CrossAttention),因为邻居之间是无序的  
        max/add/mean/attention...
    - 更新update 
        完成邻居消息聚合后，结合邻居结果与自身特征，输出最终emb 
        
    三个函数分别定义，最终通过propagate()完成对上述三个函数的计算过程
    """
    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None: 
        
        # add聚合形式 
        # default hidden params:
        # -- flow : source_to_target 
        # -- node_dim = 0 沿着0维进行传递
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        
        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim) 
        
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))  # T, emb
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int], # 当前时间步 
                edge_index: Adj,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor, # bos: beginning of seq 
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:  
        """ Agent-to-agent Interaction 
        - Agent i 在时刻t的emb feature z_i^t 
          z_i^t = MLP(Rotation_matrix(pi^t-pi^{t-1}), ai)
          
        - Agent i 在时刻t的邻接节点的emb feature z_ij^t 
          z_{ij}^t = MLP(Rotation_matrix(pj^t-pj^{t-1}), Rotation_matrix(p_j^t-p_i^t), aj) 
          a 是语义属性 
          
        - 信息聚合 Cross-Multi-head Attention  
          融合/聚合agent与周围agent的交互信息
          q_i^t = Wq*z_i^t; 自身query  
          k_{ij}^t = Wk*z_{ij}^t; 
          v_{ij}^t = Wv*z_{ij}^t 
          做cross Attention -> 
          得到自车聚合周围Agent的消息 m_i^t = \sum_{j\inN_i} (alpha_{ij}^t * v_{ij}^t)  
          
        - gate信息融合机制 
          最后使用门控Gate理念来融合交互message和agent自车信息 
          gate_i^t = sigmoid(W[z_i^t, m_i^t]) 
          z'_i^t = gate_i^t * z_i^t + (1 - gate_i^t) * m_i^t (element wise)
        
        Args:
            x (torch.Tensor): _description_
            t (Optional[int]): _description_
            edge_index (Adj): _description_
            edge_attr (torch.Tensor): _description_
            bos_mask (torch.Tensor): _description_
            rotate_mat (Optional[torch.Tensor], optional): _description_. Defaults to None.
            size (Size, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        
        # 并行? 
        if self.parallel:
            if rotate_mat is None:
                center_embed = self.center_embed(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1))
            else:
                center_embed = self.center_embed(
                    torch.matmul(x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1).unsqueeze(-2),
                                 rotate_mat.expand(self.historical_steps, *rotate_mat.shape)).squeeze(-2))
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                       self.bos_token.unsqueeze(-2),
                                       center_embed).view(x.shape[0], -1)
        else:
            if rotate_mat is None:
                center_embed = self.center_embed(x) # Agent-i在t时刻的emb
            else: 
                # AD作为center_agent无需Rotate Matrix 
                # 其他周围agent作为center_agent需要相较于AD的Rotate_matrix做变换 
                # 耦合序列开始信息
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2)) 
                
            # 这里需要mask掉当前时间步t之后的信息，这个信息是看不到的
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed) # torch.where()函数在PyTorch中用于根据提供的条件张量来从两个输入张量中选择元素。如果条件张量中的元素为True，则从第一个输入张量中选取相应位置的元素；若为False，则从第二个输入张量中选取。
        
        # self._mha_block 用于消息传递
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat, size)  
        
        # 前馈 
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor, # (N_edges, N_features)
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor: 
        # message定义消息如何从一个节点j发送给节点i 
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor: 
        # 根据聚合结果更新中心节点
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed)) # gate 
        return inputs + gate * (self.lin_self(center_embed) - inputs) # gate信息融合机制 

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor: 
        # propagate：定义消息传递的具体形式 
        # 聚合message、aggregator、update  
        # edge_index: [2, N_edges]
        # x: (N_nodes, N_features) 
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# TODO 普通实现 MLP-based,基于距离做MaxPooling
class AAEncoderMLP(MessagePassing):
    def __init__(self):
        pass 

# TODO 需要测试 (掩码部分)
class TemporalEncoder(nn.Module):
    """ 
    # Temporal Dependency 捕获每个局部区域的时间信息 
    # 每个Agent i，输入是上个模块输入的所有{s_i^T}_{t=1}^T，并加入额外可学习的Token s_{i}^{T+1}
    """
    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8, 
                 num_layers: int = 4, # 4层EncoderLayers 
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim)) # 额外可学习的Token ? 
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1) # 生成一个下三角掩码，用于 Transformer 编码器的自注意力机制， 掩码中的 -inf 表示不可关注的未来位置 
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:  
        
        # 指示序列中的填充位置 
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x) # padding mask  
        # 创建一个 cls_token 的扩展版本，并将其添加到序列的开头
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0) 
        # 残差
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor: 
        """ 
        # tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            #     [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            #     [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            #     [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
            #     [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
            #     [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
            #     [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
            #     [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
            #     [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
            #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) 
        
        mask --- zero <-> -inf  
        
        """
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor: 
        # Encoder 有skip connect 
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


# Agent-to-Lane Encoder
class ALEncoder(MessagePassing): 
    """ 
    Agent-to-lane 
    - Lane的信息可以很好的引导Center Agent未来意图的学习 
    """
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim)) # 十字路口 Intersection 
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim)) # 转弯
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim)) # 交通控制
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor,
                turn_directions: torch.Tensor,
                traffic_controls: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
                                            turn_directions, traffic_controls, rotate_mat, size)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections_j,
                turn_directions_j,
                traffic_controls_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor: 
        """ 
        先把地图信息根据Agent i的朝向旋转，每一帧都包含了旋转后的Lane的线段向量，旋转后的lane起始点相对Agent在T时刻的位置，以及地图元素的语义信息。 
        
        z_ip = mlp(Rotation_i(pp1-pp0), Rotation_i(pp0-ppT), a_p)  
        - R_i in R^(2x2) 表示Agent's i的Rotation Matrix   
        - pp -- lane的位置二元组  
        - a_p -- 属性(起始位置、结束位置、lane segment的语义属性)
        """
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                   torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]]) 
            
        # Query: Ego AD x_i(from agent-to-agent encoder <AAEncoder>)
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        
        # Key/Value: 乃是Lane的Feature 
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor: 
        """也是基于Gate来进行Agent-to-lane的融合"""
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   is_intersections: torch.Tensor,
                   turn_directions: torch.Tensor,
                   traffic_controls: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                               is_intersections=is_intersections, turn_directions=turn_directions,
                                               traffic_controls=traffic_controls, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)

if __name__ == "__main__":
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1) # 
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) 
        return mask 
    

    seq_len = 10 
    print(generate_square_subsequent_mask(seq_len))