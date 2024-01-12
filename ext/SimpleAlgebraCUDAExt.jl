module SimpleAlgebraCUDAExt
using Adapt, CUDA, FFTW, SimpleAlgebra

function Adapt.adapt_storage(::Type{CUDA.CuArray}, x::LinOpDFT) 
    Adapt.adapt_storage(CUDA.CuArray{eltype(inputspace(x))}, x)
end


function Adapt.adapt_storage(::Type{CUDA.CuArray{T}}, x::LinOpDFT) where  {T}
    dims = inputsize(x)
	#Tx = eltype(I)
    
    if T<: FFTW.fftwReal 
        forward = plan_rfft(CUDA.CuArray{T}(undef, dims))

        backward = plan_brfft(CUDA.CuArray{Complex{T}}(undef, forward.osz), dims[1];)
        outputspace = CoordinateSpace(Complex{T},forward.osz)
    else
        temp = CUDA.CuArray{T}(undef, dims)
        forward = plan_fft(temp)
        backward = plan_bfft(temp)
        outputspace = CoordinateSpace(T,forward.osz)
    end


    # Build operator.
    inputspace = CoordinateSpace(T,forward.sz)
    return LinOpDFT(inputspace, outputspace, forward, backward)

end

end