# Foundation Model for Fast Shower Simulation (FMFastSim)

## Generative Models

Different generative models can be selected by editing `gen_param' section the yaml file.

```
model_info:
    model_type:    #what kind of generative model to use
    gen_param:     #generative model paramters
    network_param: #neural network structure
```

## Auto-Encoder (AE)
AE aims to learn a low-dimensional manifold of the data. To use AE, set `variational=False` in `network_param`. Then, the reconstruction is done as

```
z_in  = model.encoding(x_in,c_in)
y_hat = model.decoding(z_in,c_in) 
```
in which `x_in` is an input variable and `c_in` is a conditioning variable.

To use AE set the following parameters:
```
model_info:
    model_type: AE
    gen_param:     
        decoder_distribution: mse    
    network_param: 
        variational: False
```
`decoder_distribution` makes it possible to train the model using a log likelihood by assuming a generative distribution. `mse` gives the standard AE. In addition, `gamma`, `normal`, `laplace`, and `cauchy` are available.


## Variational Auto-Encoder (VAE)
VAE aims to learn the data-generating distribution by using a variational inference formulation. The latent state of VAE is assumed to be a random variable with a normal distribution by setting `variational=True` in `network_param`. The reconstruction is computed as
```
z_mean,z_var = model.encoding(x_in,c_in)
z_sample     = z_mean + torch.randn_like(z_var)*z_var.sqrt()
y_hat        = model.decoding(z_sample,c_in)
```

VAE has the following parameters:
```
model_info:
    model_type: VAE
    gen_param:                  
        kl_coef: 1.0
        decoder_distribution: gamma  
        reg_coef: 0.1
        reg_model: moment_diff
    network_param: 
        variational: False
```
`kl_coef` denotes the weights of the KL divergence term in the variational lower bound (cost function)

`decoder_distribution` indicates the parameteric distribution of the decoder. There are five options: `gamma, normal, laplace, cauchy, mse`. `mse` indicates that the decoder distribution is assumed to be determinisitc.

`ref_coef` is the coefficient of a regularization

`reg_model` is what kind of regularization model to use.

## Generative Adversarial Network (GAN)
GAN tries to (implicitly) learn the data generating distribution and draw samples of the distribution. To use GAN, first disable `variational` in the `network_param`. It is important to note that GAN only uses a `decoder` part of the total network.
```
z_in  = model.prior_distribution()
y_hat = model.decoding(z_in,c_in)
```
The parameters for GAN are shown below
```
model_info:
    model_type: GAN              
    gen_param:
        gan_model: gan         
        cr_gan: 1                
        grad_norm_coef: 0
        g_net_substep: 4         
        prior_distribution: std  
        d_net_param:             
            dim_r: [18,16, 8]
            dim_a: [50,32, 8]
            dim_v: [45,32,16]
            dim_c: 4
            mlp_ratio:  3
            mlp_layers: 2
            add_filter: True     
        reg_coef: 0              
        reg_model: none          
```
`gan_model` defines the types of GAN. Use `gan` for the standard GAN and `wgan` for Wasserstein GAN.

`cr_gan: p` indicates a consistency regularization. For a consistency regularization, we first make integral transformations of the $p$-th power for the raw variable as
```math
X_r = \frac{1}{N_zN_\theta}\sum_{z,\theta} x^p_{r,z,\theta},~ X_\theta = \frac{1}{N_zN_r}\sum_{z,r} x^p_{r,z,\theta},~X_z = \frac{1}{N_\theta N_r}\sum_{\theta,r} x^p_{r,z,\theta},~X_0 = \frac{1}{N_rN_zN_\theta}\sum_{r,z,\theta}x^p_{r,z,\theta}.
```
Then, four additional discriminators are used to match the transformed variables of the generated samples with those of the data. `p=0` means `cr-gan` is disabled. It is recommended to use `cr_gan: 1` or `cr_gan: 2`.

`grad_norm_coef` is the coefficient of the gradient penalty for Wasswerstein GAN. When using `gan`, set it to zero. Setting `grad_norm_coef > 0` will slow down the computation significantly. If `wgan` is used and `grad_norm_coef: 0`, automatically a clamping is applied to make the parameter $\theta \in (-0.01,0.01)$.

`g_net_substep` is used to set the number of generator iterations per one discriminator iteration. Using CR-GAN, it is advantageous to make the number of generator steps larger than the number of discriminator steps. Typically set to `g_net_substep: 2` ~ `g_net_substep: 4`

`prior_distribution` gives an option to use a different parametric distribution for the prior of the generator. It has an option, `gamma, laplace, std`. Set it to `std`.

`d_net_param` defines the parameters of the discriminator network. Currently, a 3-D Mixer is used as a discriminator. See the code under `layers` for the details about the parameters.

In `d_net_param`, there is an item `add_filter`. When `add_filter: True`, the discriminator uses two variables as an input, the raw variable $x$ and its log transformation, $\log(x+0.001)-\log(0.001)$.

## Denoising Diffusion Model
The denoising diffusion tries to (implicitly) learn the data generating distribution and draw samples of the distribution. Again, for the denoising diffusion model, first set `variational: False` in the `network_param`. Denoising Diffusion model has the following parameters

```
model_info:
    model_type: Diffusion       
    gen_param:                  
        diff_model: orig        
        res_conn: True          
        dim_t_emb: 16           
        dim_c: 4                
        beta_info:              
            schedule: sigmoid
            num_steps: 200
            tau: 0.5
```

`diff_model` has two options; `orig` and `rescaled`. `orig` is the implementation of the original denoising diffusion model and `rescaled` is an exerperimental version. Use `orig`.

`res_conn` artificially enforces a residual connection by add the input variable at the end, i.e.
```
y_hat = x + f(x)
```
Use `res_conn: True` when the neural network does not already have a residual connection. Otherwise, set it `res_conn: False`

`dim_t_emb` defines the dimension of the positional embedding for the diffusion time step. 

`dim_c` is the dimension of the total conditioning variables, e.g., `energy(1) + angle(1) + geometry(2) = 4`.

`beta_info` defines the noise scheduling. The noise scheduling is based on `Chen, "On the Importance of Noise Scheduling for Diffusion Models," ArXiv:2301.10972v4`. It has three options: `cos, sigmoid, linear`. For the detailed parameters, see the code `diffusion.py`.
