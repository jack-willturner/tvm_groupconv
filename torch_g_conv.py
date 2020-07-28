from __future__ import absolute_import, print_function

import tvm
from tvm import te
import topi
import numpy as np
import time
from statistics import median
import torch

### COMPUTATION
N,C,X,Y = 1,16,34,34 # pretend it's already padded
K,R,S   = 16,3,3
G = 2
stride = 1
P = (X - R + 1) // stride
Q = (Y - S + 1) // stride

input_shape  = (N,C,X,Y)
weight_shape = (K,C//G,R,S)
packed_weight_shape = (G,K//G,C//G,R,S)
packed_output_shape = (N,G,K//G,P,Q)
output_shape = (N,K,P,Q)

I = te.placeholder(input_shape, name="I")
W = te.placeholder(weight_shape, name="W")

### reductions
rc = te.reduce_axis((0, C//G), name='rc')
ry = te.reduce_axis((0, R), name='ry')
rx = te.reduce_axis((0, S), name='rx')

ig = C//G
og = K//G

###  (K,C//G,R,S) to (G,K//G,C//G,R,S)
W_pack = topi.reshape(W, (packed_weight_shape))

O = te.compute(
    packed_output_shape,
    lambda n, g, co, x, y:
        te.sum(I[n,rc+(g*ig),x+rx,y+ry] * W_pack[g,co,rc,rx,ry],
            axis=[rc,ry,rx])
    )


s = te.create_schedule(O.op)
s[W_pack].compute_inline()

ir = tvm.lower(s, [I,W,O])
print(ir)

### COMPILE AND RUN
tgt_host="llvm"
tgt="llvm"
conv = tvm.build(s, [I,W,O], tgt)
ctx = tvm.context(tgt, 0)

## RUN
i = tvm.nd.array(np.random.uniform(size=input_shape).astype(np.float32), ctx)
w = tvm.nd.array(np.random.uniform(size=weight_shape).astype(np.float32), ctx)
o = tvm.nd.array(np.zeros(shape=packed_output_shape).astype(np.float32), ctx)

conv(i,w,o)

### CHECK AGAINST A TORCH CONV
o = torch.from_numpy(o.asnumpy())

i_t = torch.from_numpy(i.asnumpy())
w_t = torch.from_numpy(w.asnumpy())

conv2d = torch.nn.Conv2d(C,K,kernel_size=R,bias=False,groups=2)
conv2d.weight.data = w_t

o_t = conv2d(i_t)
o = o.reshape(1,16,32,32)
print(torch.allclose(o, o_t))
