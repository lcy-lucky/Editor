import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
DEVICE = 0
from torch import nn
import torch
import torch.nn.functional as F

def reverse_tensor(tensor=None, axis=-1):
    if tensor is None:
        return None
    if tensor.dim() <= 1:
        return tensor
    indices = range(tensor.size()[axis])[::-1]
    indices = Variable(torch.LongTensor(indices), requires_grad=False).to(tensor.device)
    return tensor.index_select(axis, indices)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=0., act=F.relu, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.attention = Parameter(torch.Tensor(32, 32))
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

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nclass, dropout):
        super(GCN, self).__init__()
        self.nodevec1 = torch.nn.Parameter(torch.randn(32, 10).to(DEVICE), requires_grad=True).to(DEVICE)
        self.nodevec2 = torch.nn.Parameter(torch.randn(10, 32).to(DEVICE), requires_grad=True).to(DEVICE)
        # torch.nn.init.xavier_uniform_(self.nodevec1)
        # torch.nn.init.xavier_uniform_(self.nodevec2)
        self.gc1 = GraphConvolution(nfeat, nhid1)
        # self.gc2 = GraphConvolution(nhid1, nhid2)
        # self.gc3 = GraphConvolution(nhid2, nhid3)
        self.gc4 = GraphConvolution(nhid1, nclass)
        self.dropout = dropout

    def forward(self, x_input, adj):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adj = adj + adp
        for i in range(x_input.shape[0]):
            if i == 0:
                gcn_output = F.relu(self.gc1(x_input[i], adj))
                gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
                # gcn_output = F.relu(self.gc2(gcn_output, adj))
                # gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
                # gcn_output = F.relu(self.gc3(gcn_output, adj))
                # gcn_output = F.dropout(gcn_output, self.dropout, training=self.training)
                gcn_output = F.relu(self.gc4(gcn_output, adj)).view([1, 32, 32])
            else:
                tmp_output = F.relu(self.gc1(x_input[i], adj))
                tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
                # tmp_output = F.relu(self.gc2(tmp_output, adj))
                # tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
                # tmp_output = F.relu(self.gc3(tmp_output, adj))
                # tmp_output = F.dropout(tmp_output, self.dropout, training=self.training)
                tmp_output = F.relu(self.gc4(tmp_output, adj)).view([1, 32, 32])
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
      
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

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

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data, gain=1.0)
        nn.init.xavier_uniform_(self.conv2.weight.data, gain=1.0)
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
        
        layers = []
        num_levels = len(num_channels)
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
    def __init__(self, gcn_nfeat, gcn_nhid1, gcn_class, gcn_dropout, 
                 tcn_num_inputs, tcn_num_channels, tcn_kernel_size, tcn_dropout):
        super(GCNTCN, self).__init__()
        
        self.GCN = GCN(nfeat=gcn_nfeat, nhid1=gcn_nhid1, nclass=gcn_class, dropout=gcn_dropout).to(DEVICE)
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
    
class bfGCNTCN(nn.Module):
    def __init__(self, gcn_nfeat, gcn_nhid1, gcn_class, gcn_dropout, 
                 tcn_num_inputs, tcn_num_channels, tcn_kernel_size, tcn_dropout):
        super(bfGCNTCN, self).__init__()
        
        self.GCNTCN_fwd = GCNTCN(gcn_nfeat, gcn_nhid1, gcn_class, gcn_dropout, 
                 tcn_num_inputs, tcn_num_channels, tcn_kernel_size, tcn_dropout).to(DEVICE)
        self.GCNTCN_bwd = GCNTCN(gcn_nfeat, gcn_nhid1, gcn_class, gcn_dropout, 
                 tcn_num_inputs, tcn_num_channels, tcn_kernel_size, tcn_dropout).to(DEVICE)
  
    def forward(self, x, x_mask, adj):
        imp_fwd = self.GCNTCN_fwd(x, x_mask, adj)
        x_bwd = reverse_tensor(x, axis=1)
        x_mask_bwd = reverse_tensor(x_mask, axis=1)
        adj_bwd = reverse_tensor(adj, axis=1)
        imp_bwd = self.GCNTCN_bwd(x_bwd, x_mask_bwd, adj_bwd)
        imp_bwd = reverse_tensor(imp_bwd, axis=1)
        # print(imp_fwd.shape)
        # print(imp_bwd.shape)
        # imputation_bf = torch.stack([imp_fwd, imp_bwd], dim=1)
        # print(imputation_bf.shape)
        imputation_bf = (imp_fwd + imp_bwd) / 2

        return imputation_bf

if __name__=='__main__':
    x = torch.randn(72,1,32).to(DEVICE)
    mask = torch.ones(72,1,32).to(DEVICE)
    adj = torch.ones(32,32).to(DEVICE)
    
