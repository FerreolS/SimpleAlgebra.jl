
function Adapt.adapt_storage(::Type{CUDA.CuArray}, x::LinOpDFT{I,O,F,B}) where  {I,O,F,B}
    dims = inputsize(x)
	T = eltype(I)
    
    if T<: fftwReal 
        forward = plan_rfft(CUDA.CuArray{T}(undef, dims))

        backward = plan_brfft(CUDA.CuArray{Complex{T}}(undef, forward.osz), dims[1];)
    else
        temp = CUDA.CuArray{T}(undef, dims)
        forward = plan_fft!(temp)
        backward = plan_bfft!(temp)
    end


    # Build operator.

	inputspace = CoordinateSpace(T,forward.sz)
	outputspace = CoordinateSpace(T,forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)

end