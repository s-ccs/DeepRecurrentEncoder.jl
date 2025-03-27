# Tutorial - How to run models on a GPU

Julia has first-class support for GPU programming. And hence, you can train the models for the DeepRecurrentEncoder directly on a GPU

To do this, just cast the data that you want to train into a GPU array class, and you're all good to go

For example, 

```julia
dre, ps, st, loss_epoch, loss_epoch_rsquared = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x-> CuArray(x),f,evts;n_epochs=25,lr=0.1,batch_size=256, hidden_chs = hidden_channels[k])
```

As you can see, the statement x -> CuArray(x) casts the array into a CudaArray (Which is an instance of a GPU array), and hence, the training of the model happens on a GPU.

Alternatively, you can just use a variable use_gpu to toggle GPU usage or not

```julia
use_gpu = true
dre, ps, st, loss_epoch, loss_epoch_rsquared = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=25,lr=0.1,batch_size=256, hidden_chs = hidden_channels[k])
```