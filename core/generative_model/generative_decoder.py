import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from core.layers.lib_mixer import Mixer2D

class Decoder_Distribution(nn.Module):
    def __init__(self,dim_r = 18,
                      dim_a = 50,
                      dim_v = 45,
                      dim_c = 10,
                      mlp_ratio=4,
                      mlp_layers=3,
                      pdf = 'gamma',  #gamma, normal, laplace, or any mixture combination,e.g.,gamma-laplace
                      norm = 'layer',
                      fix_mix=False,
        ):

        super().__init__()

        self._name = 'Decoder distribution'

        self.pdf_type = pdf

        if self.pdf_type == 'gamma':
            self.pdf_model =    Gamma(dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)
        elif self.pdf_type == 'normal':
            self.pdf_model =   Normal(dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)
        elif self.pdf_type == 'laplace':
            self.pdf_model =  Laplace(dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)
        elif self.pdf_type == 'cauchy':
            self.pdf_model =   Cauchy(dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)
        elif self.pdf_type == 'Tnormal':
            self.pdf_model = T_Normal(dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)
        elif '-' in self.pdf_type:
            mixture = self.pdf_type.split('-')
            self.pdf_model = Mixture(dim_r,dim_a,dim_v,dim_c,mlp_ratio,mlp_layers,norm=norm,mixture=mixture,fix_mix=fix_mix)
        elif self.pdf_type == 'mse':
            self.pdf_model = MSE()
        else:
            print('Error Decoder distribution is not defined')
            raise ValueError

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        return self.pdf_model(x_in,c_in)

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
    def forward(self,x_in,c_in=None):
        return x_in

class Gamma(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers,norm='layer'):
        super().__init__()
        print('Decoder is Gamma distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_a,dim_a]
        dim1 = [dim_r,dim_r]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        x0 = x_in.permute(0,3,2,1) #Batch x Vertical x Azimuthal x Radial
        param_a = self.pdf_param_a(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical

        param_a = param_a.exp()
        param_b = param_b.exp()

        self.pdf_model = distributions.Gamma(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Normal(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers,norm='layer'):
        super().__init__()
        print('Decoder is Normal distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_a,dim_a]
        dim1 = [dim_r,dim_r]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        x0 = x_in.permute(0,3,2,1) #Batch x Vertical x Azimuthal x Radial
        param_a = self.pdf_param_a(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical

        param_b = param_b.exp()+1.e-6

        self.pdf_model = distributions.Normal(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class T_Normal(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers,norm='layer'):
        super().__init__()
        print('Decoder is Truncated Normal distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_a,dim_a]
        dim1 = [dim_r,dim_r]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        x0 = x_in.permute(0,3,2,1) #Batch x Vertical x Azimuthal x Radial
        param_a = self.pdf_param_a(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical

        param_a = param_a.exp()
        param_b = param_b.exp()+1.e-4

        self.pdf_model  = [distributions.Normal( param_a,param_b)]
        self.pdf_model += [distributions.Normal(-param_a,param_b)]
        return self.pdf_model[0].rsample().abs()

    def log_prob(self,x_in):
        Prob = []
        for pdf in self.pdf_model:
            Prob += [pdf.log_prob(x_in)]
        Prob = torch.stack(Prob,dim=0)
        Log_Prob = Prob.logsumexp(dim=0)
        return Log_Prob

class Laplace(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers,norm='layer'):
        super().__init__()
        print('Decoder is Laplace distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_a,dim_a]
        dim1 = [dim_r,dim_r]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        x0 = x_in.permute(0,3,2,1) #Batch x Vertical x Azimuthal x Radial
        param_a = self.pdf_param_a(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical

        param_b = param_b.exp()+1.e-8

        self.pdf_model = distributions.Laplace(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Cauchy(nn.Module):
    def __init__(self,dim_r,dim_a,mlp_ratio,mlp_layers,norm='layer'):
        super().__init__()
        print('Decoder is Cauchy distribution')

        self.dim_r = dim_r
        self.dim_a = dim_a

        dim0 = [dim_a,dim_a]
        dim1 = [dim_r,dim_r]

        self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)
        self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        x0 = x_in.permute(0,3,2,1) #Batch x Vertical x Azimuthal x Radial
        param_a = self.pdf_param_a(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical
        param_b = self.pdf_param_b(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical

        param_b = param_b.exp()+1.e-8

        self.pdf_model = distributions.Cauchy(param_a,param_b)
        return self.pdf_model.rsample()

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Mixture(nn.Module):
    def __init__(self,dim_r,dim_a,dim_v,dim_c,mlp_ratio,mlp_layers,mixture=['normal','gamma'],fix_mix=False,norm='layer'):
        super().__init__()

        mix_dist = []
        for p in mixture:
            if p == 'gamma':
                mix_dist += [Gamma   (dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)]
            elif p == 'normal':
                mix_dist += [Normal  (dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)]
            elif p == 'laplace':
                mix_dist += [Laplace (dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)]
            elif p == 'cauchy':
                mix_dist += [Cauchy  (dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)]
            elif p == 'Tnormal':
                mix_dist += [T_Normal(dim_r,dim_a,mlp_ratio,mlp_layers,norm=norm)]
            else:
                print('wrong distribution is given: '+p)
                os.exit(-1)

        self.d0 = dim_r
        self.d1 = dim_a
        self.d2 = dim_v

        self.dim_c = dim_c

        self.fix_mix    = fix_mix
        self.mix_dist   = nn.ModuleList(mix_dist)
        self.mix_weight = nn.Sequential(nn.Linear(dim_c,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),
                                        nn.Linear(128,len(mixture)),nn.Sigmoid())

    def forward(self,x_in,c_in=None):

        if c_in == None:
            c_in = torch.zeros(x_in.size(0),self.dim_c,device=x_in.device)
        
        #self.pred_mix_weight = F.softmax(self.mix_weight(c_in),dim=-1)
        weight = self.mix_weight(c_in) + 0.2

        self.pred_mix_weight = weight/weight.sum(-1,keepdim=True)
        
        nb = x_in.size(0) #batch size
        nm = self.pred_mix_weight.size(-1) #number of mixtures

        mix_weight = self.pred_mix_weight.view(nb,1,1,1,nm).expand(-1,self.d0,self.d1,self.d2,-1)

        Mixture_Sample = distributions.Multinomial(probs=mix_weight).sample()

        x_out = 0
        for i in range(len(self.mix_dist)):
            x_out += Mixture_Sample[:,:,:,:,i]*self.mix_dist[i](x_in)

        return x_out

    def log_prob(self,x_in):

        nb = x_in.size(0) #batch size
        nm = self.pred_mix_weight.size(-1) #number of mixtures

        mix_weight = self.pred_mix_weight.view(nb,1,1,1,nm).expand(-1,self.d0,self.d1,self.d2,-1)

        #if self.fix_mix:
        #    mix_weight = mix_weight.detach()

        Prob = []
        for i in range(len(self.mix_dist)):
            Prob += [mix_weight[:,:,:,:,i].log() + self.mix_dist[i].log_prob(x_in)]
        Prob = torch.stack(Prob,dim=0)

        Log_Prob = Prob.logsumexp(dim=0)

        self.pred_mix_weight = 0 #reset predicted mixture weight

        return Log_Prob
