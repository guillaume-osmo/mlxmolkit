"""Single-dispatch float32 rotation of heavy-heavy sp integral blocks.

The 256 outputs per pair are independent. Canonical orbital indices enforce
pair symmetry without a scatter or intermediate category tensors.
"""
import mlx.core as mx

_KERNEL = None
_SOURCE = r'''
uint tid = thread_position_in_grid.x;
uint pair = tid / 256;
if (pair >= ri_shape[0]) return;
uint a = (tid / 64) % 4, b = (tid / 16) % 4;
uint c = (tid / 4) % 4, d = tid % 4;
int kk = max(a,b), ll = min(a,b), mm = max(c,d), nn = min(c,d);
int k=kk-1, l=ll-1, m=mm-1, n=nn-1;
float3 x = float3(r0[3*pair], r0[3*pair+1], r0[3*pair+2]);
float3 y = float3(r1[3*pair], r1[3*pair+1], r1[3*pair+2]);
float3 z = float3(r2[3*pair], r2[3*pair+1], r2[3*pair+2]);
device const float* I = ri + pair*22;
float v;
if (kk == 0) {
    if (mm == 0) v=I[0];
    else if (nn == 0) v=I[4]*x[m];
    else v=I[10]*x[m]*x[n]+I[11]*(y[m]*y[n]+z[m]*z[n]);
} else if (ll == 0) {
    if (mm == 0) v=I[1]*x[k];
    else if (nn == 0) v=I[5]*x[k]*x[m]+I[6]*(y[k]*y[m]+z[k]*z[m]);
    else {
        float t0=x[k]*x[m]*x[n];
        float t1=(y[m]*y[n]+z[m]*z[n])*x[k];
        float mix=y[k]*(y[n]*x[m]+y[m]*x[n])+z[k]*(z[m]*x[n]+z[n]*x[m]);
        v=I[12]*t0+I[13]*t1+I[14]*mix;
    }
} else {
    if (mm == 0) v=I[2]*x[k]*x[l]+I[3]*(y[k]*y[l]+z[k]*z[l]);
    else if (nn == 0) {
        float t0=x[k]*x[l]*x[m];
        float t1=(y[k]*y[l]+z[k]*z[l])*x[m];
        float t2a=y[l]*y[m]+z[l]*z[m];
        float t2b=y[k]*y[m]+z[k]*z[m];
        v=I[7]*t0+I[8]*t1+I[9]*(x[k]*t2a+x[l]*t2b);
    } else {
        float t0=x[k]*x[l]*x[m]*x[n];
        float t1=(y[k]*y[l]+z[k]*z[l])*x[m]*x[n];
        float t2=(y[m]*y[n]+z[m]*z[n])*x[k]*x[l];
        float quad=y[k]*y[l]*y[m]*y[n]+z[k]*z[l]*z[m]*z[n];
        float mix1=x[m]*(y[l]*y[n]+z[l]*z[n]);
        float mix2=x[n]*(y[l]*y[m]+z[l]*z[m]);
        float val5=x[k]*(mix1+mix2)+x[l]*(x[m]*(y[k]*y[n]+z[k]*z[n])+x[n]*(y[k]*y[m]+z[k]*z[m]));
        float mix3=y[k]*y[l]*z[m]*z[n]+z[k]*z[l]*y[m]*y[n];
        float cross=(y[k]*z[l]+z[k]*y[l])*(y[m]*z[n]+z[m]*y[n]);
        v=I[15]*t0+I[16]*t1+I[17]*t2+I[18]*quad+I[19]*val5+I[20]*mix3+I[21]*cross;
    }
}
w[tid]=v;
'''


def rotate_xx_batch_fused_metal(ri, r0, r1, r2):
    """Float32 MLX arrays in and out; caller explicitly opts into this precision."""
    global _KERNEL
    n = ri.shape[0]
    if n == 0:
        return mx.zeros((0, 4, 4, 4, 4), dtype=mx.float32)
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(name='nddo_rotate_xx_fused',
                    input_names=['ri', 'r0', 'r1', 'r2'], output_names=['w'], source=_SOURCE)
    return _KERNEL(inputs=[a.astype(mx.float32) for a in (ri,r0,r1,r2)],
                   output_shapes=[(n,4,4,4,4)],output_dtypes=[mx.float32],
                   grid=(n*256,1,1),threadgroup=(256,1,1))[0]
