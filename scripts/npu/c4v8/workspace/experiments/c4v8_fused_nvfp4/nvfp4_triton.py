"""Experimental Ascend Triton NVFP4 QDQ candidate. Not validated/production yet.

Preserves portable core's per-row upper scale, block16 E4M3 RNE scale, E2M1
RNE payload and FP32 operation order. No native FP4 hardware claim. Compiler
FP fusion is disabled. Scale division uses integer significands, verified by a
dedicated full-exponent probe; native tl.div_rn failed that probe on Ascend.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _float_components(x):
    x=tl.cast(x,tl.float32)
    bits=x.to(tl.int32,bitcast=True)
    exponent=(bits>>23)&255
    subnormal=exponent==0
    scaled=tl.where(subnormal,x*16777216.0,x)
    sbits=scaled.to(tl.int32,bitcast=True)
    exp=((sbits>>23)&255)-subnormal.to(tl.int32)*24
    mantissa=((sbits&0x7fffff)+0x800000).to(tl.int64)
    return exp,mantissa


@triton.jit
def _divide_positive_rne(numerator,denominator):
    a=tl.cast(numerator,tl.float32);b=tl.cast(denominator,tl.float32)
    en,mn=_float_components(a);ed,md=_float_components(b)
    ge=mn>=md
    base=tl.where(ge,23,24)
    exponent=en-ed+127-(~ge).to(tl.int32)
    shift=base+tl.minimum(exponent-1,0)
    scaled_n=mn<<tl.maximum(tl.minimum(shift,24),0).to(tl.int64)
    scaled_d=md<<tl.maximum(tl.minimum(-shift,1),0).to(tl.int64)
    q=scaled_n//scaled_d
    rem=scaled_n-q*scaled_d
    up=(2*rem>scaled_d)|((2*rem==scaled_d)&((q&1)!=0))
    rounded=q+up.to(tl.int64)
    bits=((tl.maximum(exponent,1)-1).to(tl.int64)<<23)+rounded
    bits=tl.where(exponent<-23,0,bits)
    bits=tl.where((exponent>254)|(bits>=0x7f800000),0x7f800000,bits)
    bits=tl.where(a==0,0,bits)
    bits=tl.where(b==0,0x7f800000,bits)
    invalid=(a<0)|(b<0)|(~(tl.abs(a)<float('inf')))|(~(tl.abs(b)<float('inf')))|((a==0)&(b==0))
    bits=tl.where(invalid,0x7fc00000,bits)
    return bits.to(tl.int32).to(tl.float32,bitcast=True)


@triton.jit
def _rne_integer_nonnegative(x):
    low=tl.floor(x)
    fraction=x-low
    odd=(low.to(tl.int32)&1)!=0
    return low+((fraction>0.5)|((fraction==0.5)&odd)).to(tl.float32)


@triton.jit
def _e4_positive_round(x):
    # Input is an ordinary nonnegative scale, already clamped to448.
    bits=x.to(tl.int32,bitcast=True)
    normal_bits=(bits+0x7ffff+((bits>>20)&1))&0x7ff00000
    normal=normal_bits.to(tl.float32,bitcast=True)
    subnormal=_rne_integer_nonnegative(x*512.0)*0.001953125
    return tl.minimum(tl.where(x<0.015625,subnormal,normal),448.0)


@triton.jit
def _e4_positive_code(x):
    bits=x.to(tl.int32,bitcast=True)
    normal=(((bits>>23)&255)-120)*8+((bits>>20)&7)
    return tl.where(x<0.015625,(x*512.0).to(tl.int32),normal)


@triton.jit
def _e2_code(x):
    a=tl.abs(x)
    return ((a>0.25).to(tl.int32)+(a>=0.75).to(tl.int32)+
            (a>1.25).to(tl.int32)+(a>=1.75).to(tl.int32)+
            (a>2.5).to(tl.int32)+(a>=3.5).to(tl.int32)+(a>5.0).to(tl.int32))


@triton.jit
def _e2_value(code):
    return tl.where(code<4,code*0.5,tl.where(code<6,code-2.0,(code-4.0)*2.0))


@triton.jit
def _division_probe(Numerator,Denominator,Output,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(Numerator+i,i<N,other=1.0)
    b=tl.load(Denominator+i,i<N,other=1.0)
    result=_divide_positive_rne(a,b)
    tl.store(Output+i,result,i<N)


@triton.jit
def _nvfp4_value_with_codes(x):
    # Register-only helper. Shape [ROWS,GROUPS,16], finite or explicit invalid.
    # GROUPS must include the entire logical row used for its upper scale.
    absolute=tl.abs(x)
    invalid=tl.sum(tl.sum((~(absolute<float('inf'))).to(tl.int32),2),1)>0
    block_amax=tl.max(absolute,2)
    amax=tl.max(block_amax,1)
    upper_encode=tl.minimum(_divide_positive_rne(2688.0,amax),3.4028234663852886e38)
    upper_encode=tl.where(amax==0,1.0,upper_encode)
    upper_decode=_divide_positive_rne(1.0,upper_encode)
    multiplier=upper_encode*0.1666666716337204
    scale_before=tl.minimum(block_amax*multiplier[:,None],448.0)
    scale=_e4_positive_round(scale_before)
    block_decode=scale*upper_decode[:,None]
    block_encode=tl.minimum(_divide_positive_rne(1.0,block_decode),3.4028234663852886e38)
    normalized=tl.maximum(tl.minimum(x*block_encode[:,:,None],6.0),-6.0)
    magnitude_code=_e2_code(normalized)
    magnitude=_e2_value(magnitude_code)
    dequant=(magnitude*scale[:,:,None])*upper_decode[:,None,None]
    sign=x.to(tl.int32,bitcast=True)&(-2147483648)
    result=(dequant.to(tl.int32,bitcast=True)|sign).to(tl.float32,bitcast=True)
    result=tl.where(invalid[:,None,None],float('nan'),result)
    code=magnitude_code+tl.where(sign!=0,8,0)
    return result,code,_e4_positive_code(scale),invalid


@triton.jit
def nvfp4_value_2d(x):
    """Register-only API: FP32 [ROWS,WIDTH] -> (FP32 values, invalid[ROWS]).

    WIDTH must be a power of two >=16 and represent the full logical row,
    padding missing columns with zero. No load/store, no host operations.
    This exact arithmetic is still a candidate until its NPU report passes.
    """
    rows:tl.constexpr=x.shape[0]
    width:tl.constexpr=x.shape[1]
    values,codes,scales,invalid=_nvfp4_value_with_codes(tl.reshape(x,(rows,width//16,16)))
    return tl.reshape(values,(rows,width)),invalid


@triton.jit(do_not_specialize=["NROWS"])
def _qdq(X,Y,Codes,Scales,NROWS,PHYSICAL_WIDTH:tl.constexpr,
         ROW_BLOCK:tl.constexpr,GROUP_BLOCK:tl.constexpr,STORE_BYTES:tl.constexpr):
    row=tl.program_id(0)*ROW_BLOCK+tl.arange(0,ROW_BLOCK)
    group=tl.arange(0,GROUP_BLOCK)
    lane=tl.arange(0,16)
    col=group[:,None]*16+lane[None,:]
    mask=row[:,None,None]<NROWS
    x=tl.load(X+row[:,None,None]*PHYSICAL_WIDTH+col[None,:,:],mask,other=0.0).to(tl.float32)
    result,code,scale_code,invalid=_nvfp4_value_with_codes(x)
    tl.store(Y+row[:,None,None]*PHYSICAL_WIDTH+col[None,:,:],result,mask)
    if STORE_BYTES:
        tl.store(Codes+row[:,None,None]*PHYSICAL_WIDTH+col[None,:,:],code.to(tl.uint8),mask)
        tl.store(Scales+row[:,None]*GROUP_BLOCK+group[None,:],scale_code.to(tl.uint8),row[:,None]<NROWS)


def division_probe(numerator,denominator):
    if numerator.device.type not in ('npu','privateuseone') or denominator.device!=numerator.device:
        raise ValueError('NPU-only candidate, no CPU fallback')
    a,b=torch.broadcast_tensors(numerator.float(),denominator.float())
    a,b=a.contiguous(),b.contiguous();out=torch.empty_like(a)
    _division_probe[(triton.cdiv(a.numel(),256),)](a,b,out,a.numel(),256,enable_fp_fusion=False)
    return out


def quantized_values(x,*,check=True,logical_bytes=False,row_block=4):
    if x.device.type not in ('npu','privateuseone'):
        raise ValueError('NPU-only unvalidated Triton candidate; no CPU fallback')
    if x.dtype not in (torch.float32,torch.bfloat16,torch.float16) or x.ndim<1 or x.numel()==0:
        raise ValueError('Expected nonempty floating tensor')
    if row_block not in (1,2,4,8):raise ValueError('row_block must be1/2/4/8')
    if check and not bool(torch.isfinite(x).all()):raise ValueError('NVFP4 inputs must be finite')
    x=x.contiguous();width=x.shape[-1];rows=x.numel()//width;padded=triton.cdiv(width,16)*16
    if width>4096:raise ValueError('Bounded candidate supports at most4096 columns')
    # Ascend vector copies need aligned physical rows. Padding zeros changes no
    # maxima/scales in real groups; logical byte outputs retain ceil(width/16).
    physical=max(16,triton.next_power_of_2(width))
    stored=x.reshape(rows,width)
    if physical!=width:stored=torch.nn.functional.pad(stored,(0,physical-width))
    output=torch.empty((rows,physical),device=x.device,dtype=torch.float32)
    codes=torch.empty((rows,physical),device=x.device,dtype=torch.uint8) if logical_bytes else output
    scales=torch.empty((rows,physical//16),device=x.device,dtype=torch.uint8) if logical_bytes else output
    _qdq[(triton.cdiv(rows,row_block),)](stored,output,codes,scales,rows,physical,row_block,
                                      physical//16,logical_bytes,enable_fp_fusion=False)
    output=output[:,:width].contiguous().reshape(x.shape)
    if logical_bytes:
        codes=codes[:,:padded].contiguous();scales=scales[:,:padded//16].contiguous()
    if check and not bool(torch.isfinite(output).all()):raise FloatingPointError('NVFP4 Triton output nonfinite')
    return output,(codes if logical_bytes else None),(scales if logical_bytes else None)



class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,check):return quantized_values(x,check=check)[0].to(x.dtype)
    @staticmethod
    def backward(ctx,grad_output):return grad_output,None


def nvfp4_qdq(x,*,check=True):
    return _STE.apply(x,check)
