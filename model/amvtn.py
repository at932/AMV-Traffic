import torch
import torch.nn.functional as F
import torch.nn as nn

class A_GCN(nn.Module):
    def __init__(self, input_dim, output_dim,node_num,predefined_embed_dim,unknown_embed_dim,final_embed_dim):
        super().__init__()
        self.cheb_k = 3
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        self.predefined_embed_dim = predefined_embed_dim
        self.unknown_embed_dim = unknown_embed_dim
        self.final_embed_dim = unknown_embed_dim

        #unknown embedding 未知嵌入
        self.unknown_embed = nn.Parameter(torch.FloatTensor(node_num,unknown_embed_dim))

        #embedding转换全连接层获取最终嵌入
        self.final_embed_linear = nn.Linear(predefined_embed_dim+unknown_embed_dim,final_embed_dim)

        #gcn参数
        self.weights_pool = nn.Parameter(torch.FloatTensor(final_embed_dim,self.cheb_k,input_dim,output_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(final_embed_dim,output_dim))

        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.kaiming_normal_(self.unknown_embed, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.weights_pool, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.bias_pool, mode='fan_in', nonlinearity='relu')

    def forward(self,x,predefined_embed):
        batch_size,_,_ = x.shape #[B,N,1]
        embed = torch.cat((predefined_embed,self.unknown_embed.unsqueeze(0).repeat_interleave(batch_size,dim=0)),dim=2) #[B,N,predefined_dim+unknown_dim]
        final_embed = self.final_embed_linear(embed)#[B,N,final_embed_dim]


        supports = F.softmax(F.relu(torch.bmm(final_embed, final_embed.permute(0,2,1))), dim=2)#[B,N,N]
        IN = torch.eye(self.node_num)
        batched_IN = IN.unsqueeze(0).repeat_interleave(batch_size,dim=0)
        support_set = [batched_IN.to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.bmm(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=1) #[B,cheb_k,N,N]

        #参数空间化，即每个node一个参数
        # [B,N,final_embed_dim],[final_embed_dim,cheb_k,in_dim,out_dim] -> [B,N,cheb_k,in_dim,out_dim]
        weights = torch.einsum('bnd,dkio->bnkio', final_embed, self.weights_pool)
        # [B,N,final_embed_dim],[final_embed_dim,out_dim] -> [B,N,out_dim]
        bias = torch.einsum('bnd,do->bno',final_embed, self.bias_pool)
        #[B,cheb_k,N,N],[B,N,in_dim] -> [B,cheb_k,N,in_dim]
        x_g = torch.einsum("bknn,bni->bkni", supports, x)
        #[B,cheb_k,N,in_dim]->[B,N,cheb_k,in_dim]
        x_g = x_g.permute(0, 2, 1, 3)
        #[B,N,cheb_k,in_dim],[B,N,ched_k,in_dim,out_dim]->[B,N,output_dim]
        x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias

        return x_gconv


class A_MVGCN(nn.Module):
    def __init__(self,input_dim,output_dim,node_num,node_embed_dim,td_embed_dim,dw_embed_dim,envs_embed_dim,speed_past_dim,speed_future_dim,unknown_embed_dim,final_embed_dim,gcn_hidden_dim):
        super().__init__()
        self.input_dim = input_dim #输入特征维度，目前数据集为1，即仅有traffic speed一种类型
        self.output_dim = output_dim #输出特征维度，也为1
        self.gcn_hidden_dim = gcn_hidden_dim #gcn隐藏特征维度
        self.node_num = node_num #道路数量

        #spatial view的先验特征相关参数
        self.node_embed_dim =node_embed_dim #node2vec向量维度

        #environment view的先验特征相关参数
        self.td_embed_dim = td_embed_dim #time of day向量维度
        self.dw_embed_dim = dw_embed_dim #day of week向量维度
        self.envs_embed_dim = envs_embed_dim #环境特征向量维度
        #temporal view的先验特征相关参数
        self.speed_past_dim = speed_past_dim #估计时刻往前邻近时刻的长度
        self.speed_future_dim = speed_future_dim #估计时刻往后邻近时刻的长度

        self.unknown_embed_dim = unknown_embed_dim #未观测特征维度
        self.final_embed_dim = unknown_embed_dim #观测+未观测特征合并后的特征维度


        #spatial veiw
        self.spatial_a_gcn = A_GCN(input_dim=input_dim, output_dim=gcn_hidden_dim,node_num=node_num,predefined_embed_dim=node_embed_dim,unknown_embed_dim=unknown_embed_dim,final_embed_dim=final_embed_dim)
        #environment view
        self.environment_a_gcn = A_GCN(input_dim=input_dim,output_dim=gcn_hidden_dim,node_num=node_num,predefined_embed_dim=td_embed_dim+dw_embed_dim+envs_embed_dim,unknown_embed_dim=unknown_embed_dim,final_embed_dim=final_embed_dim)
        #temproal view
        self.temporal_a_gcn = A_GCN(input_dim=input_dim,output_dim=gcn_hidden_dim,node_num=node_num,predefined_embed_dim=speed_past_dim+speed_future_dim,unknown_embed_dim=unknown_embed_dim,final_embed_dim=final_embed_dim)
        # 输出全连接层
        self.out_fc = nn.Linear(gcn_hidden_dim, output_dim)

        #多视角融合参数
        self.weight_num = 3
        self.weight_init_value = 1 / self.weight_num
        self.graph_weight = nn.Parameter(torch.FloatTensor(self.weight_num))
        self.softmax = nn.Softmax(dim=0)

        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.constant_(self.graph_weight, self.weight_init_value)

    def forward(self,x,embed):
        #x [B,N]
        #embed[b,n,dim]
        batch_size,_ = x.shape
        x = x.unsqueeze(2) #[B,N,1]
        node_embed,td_embed,dw_embed,envs_embed,speed_past_embed,speed_future_embed = torch.split(embed,[self.node_embed_dim,self.td_embed_dim,self.dw_embed_dim,self.envs_embed_dim,self.speed_past_dim,self.speed_future_dim],dim=2)

        spatial_pretrained_embed = node_embed
        environment_pretrained_embed = torch.cat([td_embed,dw_embed,envs_embed],dim=2)
        temporal_pretrained_embed = torch.cat([speed_past_embed,speed_future_embed],dim=2)

        #print("spatial agcn")
        spatial_out = self.spatial_a_gcn(x,spatial_pretrained_embed) #gcn1
        #print("temporal agcn")
        environment_out = self.environment_a_gcn(x,environment_pretrained_embed) #gcn2
        #print("data agcn")
        temporal_out = self.temporal_a_gcn(x,temporal_pretrained_embed) #gcn3

        w = self.softmax(self.graph_weight)
        out = F.relu(w[0] * spatial_out + w[1] * environment_out + w[2]*temporal_out)

        out = self.out_fc(out)  # 全连接层
        out = out.squeeze(2)

        return out
