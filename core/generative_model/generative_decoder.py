import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from core.layers.lib_mixer import Mixer2D,build_mixer_block

class Decoder_Distribution(nn.Module):
    def __init__(self,dim_r = 18,
                      dim_a = 50,
                      dim_v = 45,
                      dim_c = 10,
                      mlp_ratio=4,
                      mlp_layers=3,
                      pdf = 'gamma',  #gamma, normal, laplace, or any mixture combination,e.g.,gamma-laplace
                      norm = 'layer',
                      dec_type = 'mixer',
                      fix_mix=False,
        ):

        super().__init__()

        self._name = 'Decoder distribution'

        self.pdf_type = pdf

        param_in = {'dim_r':dim_r,
                    'dim_a':dim_a,
                    'dim_v':dim_v,
                    'mlp_ratio':mlp_ratio,
                    'mlp_layers':mlp_layers,
                    'norm':norm,
                    'dec_type':dec_type}

        if self.pdf_type == 'gamma':
            self.pdf_model =    Gamma(**param_in)
        elif self.pdf_type == 'normal':
            self.pdf_model =   Normal(**param_in)
        elif self.pdf_type == 'laplace':
            self.pdf_model =  Laplace(**param_in)
        elif self.pdf_type == 'cauchy':
            self.pdf_model =   Cauchy(**param_in)
        elif self.pdf_type == 'Tnormal':
            self.pdf_model = T_Normal(**param_in)
        elif '-' in self.pdf_type:
            mixture = self.pdf_type.split('-')

            param_mix = {'dim_c':dim_c,
                         'mixture':mixture,
                         'fix_mix':fix_mix}

            self.pdf_model = Mixture(dist_param=param_in,mix_param=param_mix)
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

class Two_Param_PDF(nn.Module):
    def __init__(self,dim_r,dim_a,dim_v,mlp_ratio,mlp_layers,norm='layer',dec_type='mixer'):
        super().__init__()
    
        self.dim_r = dim_r
        self.dim_a = dim_a
        self.dim_v = dim_v

        self.dec_type = dec_type

        dim0 = [dim_a,dim_a]
        dim1 = [dim_r,dim_r]

        if self.dec_type == 'mixer':
            self.pdf_param_a = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)
            self.pdf_param_b = Mixer2D(dim0,dim1,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,norm=norm)
        elif self.dec_type == 'conv':
            norm_dim = [dim_a,dim_r,dim_v]
            self.pdf_param = nn.Sequential(nn.LayerNorm(norm_dim,elementwise_affine=False, bias=False),
                                           nn.Conv2d(  dim_a,4*dim_a,3,padding='same'),nn.SiLU(),
                                           nn.Conv2d(4*dim_a,4*dim_a,3,padding='same'),nn.SiLU(),
                                           nn.Conv2d(4*dim_a,2*dim_a,1))
                                           

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in):
        if self.dec_type == 'mixer':
            x0 = x_in.permute(0,3,2,1) #Batch x Vertical x Azimuthal x Radial
            self.param_a = self.pdf_param_a(x0).permute(0,3,2,1) #Batch x Radial x Azimuthal x Vertical
            self.param_b = self.pdf_param_b(x0).permute(0,3,2,1)
        elif self.dec_type == 'conv':
            x0 = x_in.permute(0,2,1,3) #Batch x Azimuthal x  Radial x Vertical
            param = self.pdf_param(x0)
            self.param_a = param[:,:self.dim_a,:,:].permute(0,2,1,3) #Batch x Radial x Azimuthal x Vertical
            self.param_b = param[:,self.dim_a:,:,:].permute(0,2,1,3) #Batch x Radial x Azimuthal x Vertical

    def log_prob(self,x_in):
        return self.pdf_model.log_prob(x_in)

class Gamma(Two_Param_PDF):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print('Decoder is Gamma distribution')

    def forward(self,x_in,c_in=None):
        super().forward(x_in)

        self.param_a = self.param_a.exp()
        self.param_b = self.param_b.exp()

        self.pdf_model = distributions.Gamma(self.param_a,self.param_b)
        return self.pdf_model.rsample()

class Normal(Two_Param_PDF):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print('Decoder is Normal distribution')

    def forward(self,x_in,c_in=None):
        super().forward(x_in)

        self.param_b = self.param_b.exp()+1.e-6

        self.pdf_model = distributions.Normal(self.param_a,self.param_b)
        return self.pdf_model.rsample()

class T_Normal(Two_Param_PDF):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print('Decoder is Truncated Normal distribution')

    def forward(self,x_in,c_in=None):
        super().forward(x_in)

        self.param_a = self.param_a.exp()
        self.param_b = self.param_b.exp()+1.e-4

        self.pdf_model  = [distributions.Normal( self.param_a,self.param_b)]
        self.pdf_model += [distributions.Normal(-self.param_a,self.param_b)]
        return self.pdf_model[0].rsample().abs()

    def log_prob(self,x_in):
        Prob = []
        for pdf in self.pdf_model:
            Prob += [pdf.log_prob(x_in)]
        Prob = torch.stack(Prob,dim=0)
        Log_Prob = Prob.logsumexp(dim=0)
        return Log_Prob

class Laplace(Two_Param_PDF):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print('Decoder is Laplace distribution')

    def forward(self,x_in,c_in=None):
        super().forward(x_in)

        self.param_b = self.param_b.exp()+1.e-8

        self.pdf_model = distributions.Laplace(self.param_a,self.param_b)
        return self.pdf_model.rsample()

class Cauchy(Two_Param_PDF):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print('Decoder is Cauchy distribution')

    def forward(self,x_in,c_in=None):
        super().forward(x_in)

        self.param_b = self.param_b.exp()+1.e-8

        self.pdf_model = distributions.Cauchy(self.param_a,self.param_b)
        return self.pdf_model.rsample()

class Mixture(nn.Module):
    def __init__(self,dist_param,mix_param):
        super().__init__()

        mix_dist = []
        for p in mix_param['mixture']:
            if p == 'gamma':
                mix_dist += [Gamma   (**dist_param)]
            elif p == 'normal':
                mix_dist += [Normal  (**dist_param)]
            elif p == 'laplace':
                mix_dist += [Laplace (**dist_param)]
            elif p == 'cauchy':
                mix_dist += [Cauchy  (**dist_param)]
            elif p == 'Tnormal':
                mix_dist += [T_Normal(**dist_param)]
            else:
                print('wrong distribution is given: '+p)
                os.exit(-1)

        self.d0 = dist_param['dim_r']
        self.d1 = dist_param['dim_a']
        self.d2 = dist_param['dim_v']

        self.dim_c = mix_param['dim_c']

        self.fix_mix    = mix_param['fix_mix']
        self.mix_dist   = nn.ModuleList(mix_dist)
        self.mix_weight = nn.Sequential(nn.Linear(self.dim_c,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),
                                        nn.Linear(128,len(mix_dist)),nn.Sigmoid())

    def forward(self,x_in,c_in=None):

        if c_in == None:
            c_in = torch.zeros(x_in.size(0),self.dim_c,device=x_in.device)
        
        #self.pred_mix_weight = F.softmax(self.mix_weight(c_in),dim=-1)
        weight = self.mix_weight(c_in) + 0.3

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

        nb = self.pred_mix_weight.size(0) #batch size
        nm = self.pred_mix_weight.size(1) #number of mixtures

        log_mix_weight = self.pred_mix_weight.log()
        log_mix_weight = log_mix_weight.t().view(nm,nb,1,1,1).expand(-1,-1,self.d0,self.d1,self.d2)

        #if self.fix_mix:
        #    mix_weight = mix_weight.detach()

        Prob = []
        for i in range(len(self.mix_dist)):
            Prob += [log_mix_weight[i,:,:,:,:] + self.mix_dist[i].log_prob(x_in)]
        Prob = torch.stack(Prob,dim=0)

        Log_Prob = Prob.logsumexp(dim=0)

        self.pred_mix_weight = 0 #reset predicted mixture weight

        return Log_Prob
