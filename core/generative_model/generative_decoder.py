import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from core.layers.lib_mixer import Mixer2D

class Decoder_Distribution(nn.Module):
    def __init__(self,dim_r = 18,
                      dim_a = 50,
                      dim_v = 45,
                      mlp_ratio=4,
                      mlp_layers=4,
                      pdf = 'gamma',  #gamma, normal, laplace, or any mixture combination,e.g.,gamma-laplace
                      fix_mix=False,
        ):

        super().__init__()

        self._name = 'Decoder distribution'

        self.pdf_type = pdf

        if self.pdf_type == 'gamma':
            self.pdf_model = Gamma(dim_r,dim_a,mlp_ratio,mlp_layers)
        elif self.pdf_type == 'normal':
            self.pdf_model = Normal(dim_r,dim_a,mlp_ratio,mlp_layers)
        elif self.pdf_type == 'laplace':
            self.pdf_model = Laplace(dim_r,dim_a,mlp_ratio,mlp_layers)
        elif self.pdf_type == 'cauchy':
            self.pdf_model = Cauchy(dim_r,dim_a,mlp_ratio,mlp_layers)
        elif '-' in self.pdf_type:
            mixture = self.pdf_type.split('-')
            self.pdf_model = Mixture(dim_r,dim_a,dim_v,mlp_ratio,mlp_layers,mixture=mixture,fix_mix=fix_mix)
        elif self.pdf_type == 'mse':
            self.pdf_model = MSE()
        else:
            print('Error Decoder distribution is not defined')
            raise ValueError

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in):
        return self.pdf_model(x_in)

    def Loss(self,y_hat=None,y_true=None):
        if self.pdf_type == 'mse':
            loss = (y_hat-y_true).pow(2).mean()
        else:
            loss = -self.pdf_model.log_prob(y_true).mean()
        return loss

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        print('Decoder is deterministic')

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in):
        return x_in

class Gamma(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers):
        super().__init__()
        print('Decoder is Gamma distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_r,dim_r]
        dim1 = [dim_a,dim_a]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in):
        x0 = x_in.permute(0,3,1,2) #Batch x Vertical x Radial x Azimuthal
        param_a = self.pdf_param_a(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical

        param_a = param_a.exp()
        param_b = param_b.exp()

        self.pdf_model = distributions.Gamma(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Normal(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers):
        super().__init__()
        print('Decoder is Normal distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_r,dim_r]
        dim1 = [dim_a,dim_a]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in):
        x0 = x_in.permute(0,3,1,2) #Batch x Vertical x Radial x Azimuthal
        param_a = self.pdf_param_a(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical

        param_b = param_b.exp()+1.e-6

        self.pdf_model = distributions.Normal(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Laplace(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers):
        super().__init__()
        print('Decoder is Laplace distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_r,dim_r]
        dim1 = [dim_a,dim_a]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in):
        x0 = x_in.permute(0,3,1,2) #Batch x Vertical x Radial x Azimuthal
        param_a = self.pdf_param_a(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical

        param_b = param_b.exp()+1.e-8

        self.pdf_model = distributions.Laplace(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Cauchy(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers):
        super().__init__()
        print('Decoder is Cauchy distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_r,dim_r]
        dim1 = [dim_a,dim_a]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=False,gated_attn=False)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in):
        x0 = x_in.permute(0,3,1,2) #Batch x Vertical x Radial x Azimuthal
        param_a = self.pdf_param_a(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical

        param_b = param_b.exp()+1.e-8

        self.pdf_model = distributions.Cauchy(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Mixture(nn.Module):
    def __init__(self,dim_r,dim_a,dim_v,mlp_ratio,mlp_layer,mixture=['normal','gamma'],fix_mix=False):
        super().__init__()

        mix_dist = []
        for p in mixture:
            if p == 'gamma':
                mix_dist += [Gamma  (dim_r,dim_a,mlp_ratio,mlp_layers)]
            elif p == 'normal':
                mix_dist += [Normal (dim_r,dim_a,mlp_ratio,mlp_layers)]
            elif p == 'laplace':
                mix_dist += [Laplace(dim_r,dim_a,mlp_ratio,mlp_layers)]
            elif p == 'cauchy':
                mix_dist += [Cauchy (dim_r,dim_a,mlp_ratio,mlp_layers)]
            else:
                print('wrong distribution is given: '+p)
                os.exit(-1)

        d0 = dim_r
        d1 = dim_a
        d2 = dim_v

        self.fix_mix    = fix_mix
        self.mix_dist   = nn.ModuleList(mix_dist)
        self.mix_weight = nn.Parameter(torch.ones(d0,d1,d2,len(mixture),requires_grad=True))

    def forward(self,x_in):

        mix_weight = F.softmax(self.mix_weight.repeat(x_in.size(0),1,1,1,1),dim=-1)

        Mixture_Sample = distributions.Multinomial(probs=mix_weight).sample()

        x_out = 0
        for i in range(len(self.mix_dist)):
            x_out += Mixture_Sample[:,:,:,:,i]*self.mix_dist[i](x_in)

        return x_out

    def log_prob(self,x_in):

        mix_weight = F.softmax(self.mix_weight,-1)
        if self.fix_mix:
            mix_weight = mix_weight.detach()

        Prob = []
        for i in range(len(self.mix_dist)):
            Prob += [mix_weight[:,:,:,i].log() + self.mix_dist[i].log_prob(x_in)]
        Prob = torch.stack(Prob,dim=0)

        Log_Prob = Prob.logsumexp(dim=0)

        return Log_Prob
