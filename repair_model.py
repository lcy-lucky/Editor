import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
DEVICE = 0
# from torch_geometric.nn import GATConv
# from torch_geometric.nn import GATConv
from torch import nn
import torch
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.attention = Parameter(torch.Tensor(12, 12))
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
            
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.attention)
        # torch.nn.init.kaiming_uniform_(self.weight)
        # torch.nn.init.kaiming_uniform_(self.attention)
        if self.use_bias:
            # 零值初始化
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(self.attention * adj, support)
        # output = torch.spmm(adj, support)
        # output = self.act(output)
        if self.use_bias:
            output += self.bias
        # output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
#         super(GCN, self).__init__()
#         self.nodevec1 = torch.nn.Parameter(torch.randn(12, 10).to(DEVICE), requires_grad=True).to(DEVICE)
#         self.nodevec2 = torch.nn.Parameter(torch.randn(10, 12).to(DEVICE), requires_grad=True).to(DEVICE)
#         # torch.nn.init.xavier_uniform_(self.nodevec1)
#         # torch.nn.init.xavier_uniform_(self.nodevec2)
#         self.gc1 = GraphConvolution(nfeat, nhid1)
#         self.gc2 = GraphConvolution(nhid1, nhid2)
#         self.gc3 = GraphConvolution(nhid2, nclass)
#         self.dropout = dropout

#     def forward(self, x_input, adj):
#         adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#         adj = adj + adp
#         for i in range(x_input.shape[0]):
#             if i == 0:
#                 gcn_output = F.relu(self.gc1(x_input[i], adj))
#                 gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
#                 gcn_output = F.relu(self.gc2(gcn_output, adj))
#                 gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
#                 gcn_output = F.relu(self.gc3(gcn_output, adj)).view([1, 12, 40])
#             else:
#                 tmp_output = F.relu(self.gc1(x_input[i], adj))
#                 tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
#                 tmp_output = F.relu(self.gc2(tmp_output, adj))
#                 tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
#                 tmp_output = F.relu(self.gc3(tmp_output, adj)).view([1, 12, 40])
#                 gcn_output = torch.cat((gcn_output, tmp_output),dim=0)
#         return gcn_output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass, dropout):
        super(GCN, self).__init__()
        self.nodevec1 = torch.nn.Parameter(torch.randn(12, 10).to(DEVICE), requires_grad=True).to(DEVICE)
        self.nodevec2 = torch.nn.Parameter(torch.randn(10, 12).to(DEVICE), requires_grad=True).to(DEVICE)
        # torch.nn.init.xavier_uniform_(self.nodevec1)
        # torch.nn.init.xavier_uniform_(self.nodevec2)
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nhid3)
        self.gc4 = GraphConvolution(nhid3, nclass)
        self.dropout = dropout

    def forward(self, x_input, adj):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adj = adj + adp
        for i in range(x_input.shape[0]):
            if i == 0:
                gcn_output = F.relu(self.gc1(x_input[i], adj))
                gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
                gcn_output = F.relu(self.gc2(gcn_output, adj))
                gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
                gcn_output = F.relu(self.gc3(gcn_output, adj))
                gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
                gcn_output = F.relu(self.gc4(gcn_output, adj)).view([1, 12, 40])
            else:
                tmp_output = F.relu(self.gc1(x_input[i], adj))
                tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
                tmp_output = F.relu(self.gc2(tmp_output, adj))
                tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
                tmp_output = F.relu(self.gc3(tmp_output, adj))
                tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
                tmp_output = F.relu(self.gc4(tmp_output, adj)).view([1, 12, 40])
                gcn_output = torch.cat((gcn_output, tmp_output),dim=0)
        return gcn_output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        """
        构成TCN的核心Block, 原作者在图中成为Residual block, 是因为它存在残差连接.
        但注意, 这个模块包含了2个Conv1d.

        :param n_inputs: int, 输入通道数或者特征数
        :param n_outputs: int, 输出通道数或者特征数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长, 在TCN固定为1
        :param dilation: int, 膨胀系数. 与这个Residual block(或者说, 隐藏层)所在的层数有关系. 
                                例如, 如果这个Residual block在第1层, dilation = 2**0 = 1;
                                      如果这个Residual block在第2层, dilation = 2**1 = 2;
                                      如果这个Residual block在第3层, dilation = 2**2 = 4;
                                      如果这个Residual block在第4层, dilation = 2**3 = 8 ......
        :param padding: int, 填充系数. 与kernel_size和dilation有关. 
        :param dropout: float, dropout比率
        """
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
        # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # 在整个Residual block中有非线性的激活. 这个容易忽略!
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data, gain=1.0)# xavier初始化
        nn.init.xavier_uniform_(self.conv2.weight.data, gain=1.0)# xavier初始化
        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            # self.downsample.weight.data.normal_(0, 0.01)
            nn.init.xavier_uniform_(self.downsample.weight.data, gain=1.0)# xavier初始化
        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        # if self.downsample is not None:
        #     self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        """
        :param num_inputs: int,  输入通道数或者特征数
        :param num_channels: list, 每层的hidden_channel数. 例如[5,12,3], 代表有3个block, 
                                block1的输出channel数量为5; 
                                block2的输出channel数量为12;
                                block3的输出channel数量为3.
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        layers = []
        num_levels = len(num_channels)
		# 可见，如果num_channels=[5,12,3]，那么
		# block1的dilation_size=1
		# block2的dilation_size=2
		# block3的dilation_size=4
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GCNTCN(nn.Module):
    def __init__(self, gcn_nfeat, gcn_nhid1, gcn_nhid2, gcn_nhid3, gcn_class, gcn_dropout, 
                 tcn_num_inputs, tcn_num_channels, tcn_kernel_size, tcn_dropout):
        super(GCNTCN, self).__init__()
        
        self.GCN = GCN(nfeat=gcn_nfeat, nhid1=gcn_nhid1, nhid2=gcn_nhid2, nhid3=gcn_nhid3,
                                       nclass=gcn_class, dropout=gcn_dropout).to(DEVICE)
        self.tcn = TemporalConvNet(num_inputs=tcn_num_inputs, num_channels=tcn_num_channels, 
                                   kernel_size=tcn_kernel_size, dropout=tcn_dropout).to(DEVICE)
  
    def forward(self, x, x_mask, adj):
        x_input = torch.zeros_like(x)
        mid_x_input = torch.zeros_like(x)
        x_input = torch.mul(x, x_mask)
        mid_x_input = torch.mul(x, x_mask)
        imputation_x = torch.zeros_like(x)
        
        for i in range(x_input.shape[1]):
            if i == 0:
                x_input = x_input.view([x_input.shape[0], x_input.shape[2], x_input.shape[1]])
                tcn_output = self.tcn(x_input)
                tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
                tcn_output = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask)    
            else:
                x_input = mid_x_input.view([mid_x_input.shape[0], mid_x_input.shape[2], mid_x_input.shape[1]])
                tcn_output = self.tcn(x_input)
                tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
                tcn_output = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask)
            
            w = torch.ones_like(x)
            w[:,i,:] = 0
            
            v = torch.zeros_like(x)
            v[:,i,:] = 1
            
            imputation_x = torch.mul(imputation_x,w) + torch.mul(tcn_output,v)
            mid_x_input = torch.mul(mid_x_input,w) + torch.mul(tcn_output,v)
                   

        gat_output = self.GCN(imputation_x, adj)
        # gat_output = gat_output.view([gat_output.shape[0], gat_output.shape[2], gat_output.shape[1]])
        gat_output = torch.mul(x, x_mask)+torch.mul(gat_output, 1-x_mask)        
        # tcn_output = self.tcn(x_input)
        # tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
        # decode_output = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask)
        return gat_output
    

    def forward_one_tcn(self, x, x_mask, adj):
        # print(x.shape)
        x_input = torch.zeros_like(x)
        mid_x_input = torch.zeros_like(x)
        # print(x_input.shape) # [36,1,40]
        x_input = torch.mul(x, x_mask)
        # x_input = torch.mul(x, x_mask)+rand_number
        mid_x_input = torch.mul(x, x_mask)
        
        imputation_x = torch.zeros_like(x)
        
        
        for i in range(x_input.shape[1]):
            if i == 0:
                x_input = x_input.view([x_input.shape[0], x_input.shape[2], x_input.shape[1]])
                tcn_output = self.tcn(x_input)
                tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
                tcn_output = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask)
            else:
                x_input = mid_x_input.view([mid_x_input.shape[0], mid_x_input.shape[2], mid_x_input.shape[1]])
                tcn_output = self.tcn(x_input)
                tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
                tcn_output = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask)
            
            
            # imputation_x[:,i,:].data.copy_(tcn_output[:,i,:].data)
            # mid_x_input[:,i,:].data.copy_(tcn_output[:,i,:].data)
            w = torch.ones_like(x)
            w[:,i,:] = 0
            
            v = torch.zeros_like(x)
            v[:,i,:] = 1
            
            imputation_x = torch.mul(imputation_x,w) + torch.mul(tcn_output,v)
            mid_x_input = torch.mul(mid_x_input,w) + torch.mul(tcn_output,v)
            
            # imputation_x[:,i,:] = tcn_output[:,i,:]
            # mid_x_input[:,i,:] = tcn_output[:,i,:]
            # x_input = x_input.view([x_input.shape[0], x_input.shape[2], x_input.shape[1]])
            # tcn_output = self.tcn(x_input)
            # tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
            # x_input = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask)
                
        # tcn_output = self.tcn(x_input)
        # tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
        # decode_output = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask)
        return imputation_x
        # print(x_input.shape) # [36,40,1]
        # gat_output = self.transformers_encode(x_input, adj)
        # print(gat_output.shape) # [36,40,1]
        
        # gat_output = gat_output.view([gat_output.shape[0], gat_output.shape[2], gat_output.shape[1]])
        # gat_output = torch.mul(x, x_mask)+torch.mul(gat_output, 1-x_mask)
        # gat_output = gat_output.view([gat_output.shape[0], gat_output.shape[2], gat_output.shape[1]])
        # return gat_output
        
        
        # # print(batchsize, self.tran_step)
        # tcn_input = gat_output.view([int(gat_output.shape[0]/self.tran_step), gat_output.shape[1], self.tran_step])
        # # print(tcn_input.shape) # [3,40,12]
        
        # for i in range(tcn_input.shape[1]):
            
        #     if i==0:
        #         tcn_output = self.tcn[i](tcn_input[:,i:i+1,:])
        #     else:
        #         tcn_output = torch.cat((tcn_output, self.tcn[i](tcn_input[:,i:i+1,:])), dim=1)
        
        
        # 第一次插补
        # tcn_output = self.tcn(tcn_input)
        # print(tcn_output.shape) # [3,40,12]
        # tcn_output = tcn_output.view([int(tcn_output.shape[0]*tcn_output.shape[2]), 1, tcn_output.shape[1]])
        # tcn_output = torch.mul(x, x_mask)+torch.mul(tcn_output, 1-x_mask) 
        # print(tcn_output.shape) # [36,1,40]
        # return tcn_output
        # tcn_output = tcn_output.view([tcn_output.shape[0], tcn_output.shape[2], tcn_output.shape[1]])
        # # print(tcn_output.shape)
        # decode_output = self.transformers_decode(tcn_output, adj)
        # # print(decode_output.shape)
       
        # # 第二次插补
        # decode_output = decode_output.view([decode_output.shape[0], decode_output.shape[2], decode_output.shape[1]])
        # decode_output = torch.mul(x, x_mask)+torch.mul(decode_output, 1-x_mask)
        # # decode_output = torch.mul(x, x_mask)
        # # print('2',decode_output.shape)
        # return decode_output



if __name__=='__main__':
    x = torch.randn(72,1,40).to(DEVICE)
    mask = torch.ones(72,1,40).to(DEVICE)
    adj = torch.ones(40,40).to(DEVICE)
    # gat = GAT(1,1,1,0.1,1,2)
    # trantcn = TransfomerTCN(tran_nfeat=1, tran_nhid=1, tran_class=1, tran_dropout=0.2, tran_alpha=0.2, tran_nheads=5, tran_step=12, tcn_num_inputs=1, tcn_num_channels=[2,4,8,4,2,1], tcn_kernel_size=3, tcn_dropout=0.3).to(DEVICE)
    # print(trantcn(x,mask,adj).shape)
    


