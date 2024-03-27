# Foundation Model for Fast Shower Simulation (FMFastSim)

## model implementation
A neural network model is assumed to have an encoder-decoder structure. To implement a new model, a neural network should be broken into two parts such that the final model outcome is computed as

```
z_in  = model.encoding(x_in,c_in)
y_hat = model.decoding(z_in,c_in) 
```
in which `x_in` is an input variable and `c_in` is a conditioning variable. 

A template class `layer.py` is provided for an instruction. Besides the parameters specific to each models, `layer` class should receive `variational` as one of the input arguments, which is `False` by default. If `variational = True`, the output of `encoding` function should be two tensors for mean and variance of a normal distribution, such that

```
z_mean, z_var = model.encoding(x_in,c_in)
z_in  = z_mean + torch.randn_like(z_var)*z_var.sqrt()
y_hat = model.decoding(z_in,c_in)
```
which is equivalent to
```
z_in  = model.encoding(x_in,c_in,sampling=True)
y_hat = model.decoding(z_in,c_in) 
```

During the model construction, the layer class should make an empty tensor `self.decoder_input`, which is size of `(1,input dimension of decoder)`. It can be generated simply
```
class new_model(layer):
    def __init__(self,dim_r=18,dim_a=50,dim_v=45,...):  #radial, azimuthal, vertical dimensions
        super().__init__()
        self.encoder = define_encoder(...)
        self.decoder = define_decoder(...)
        
        x_in = torch.zeros(1,dim_r,dim_a,dim_v)
        self.decoder_input = self.encoder(x_in).copy()
```
`self.decoder_input` is used to generate a prior distribution for GAN.
