
function Adapt.adapt_storage(::Type{CuArray}, x::LinOpDFT) 
    dims = inputsize(x)
	T = eltype(inputspace(x))
    planning = planning = check_flags(FFTW.MEASURE)
    timelimit = FFTW.NO_TIMELIMIT
    # Compute the plans with suitable FFTW flags.  For maximum efficiency, the
    # transforms are always applied in-place and thus cannot preserve their
    # inputs.
    
    if T<: fftwReal 
        forward = plan_rfft(CUDA.CuArray{T}(undef, dims);
                        flags = (planning | FFTW.PRESERVE_INPUT),
                        timelimit = timelimit)

        backward = plan_brfft(CUDA.CuArray{Complex{T}}(undef, forward.osz), dims[1];
                          flags = (planning | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)
    else
        temp = CUDA.CuArray{T}(undef, dims)
        forward = plan_fft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                        timelimit = timelimit)
        backward = plan_bfft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)
    end


    # Build operator.

	inputspace = CoordinateSpace(T,forward.sz)
	outputspace = CoordinateSpace(T,forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)

end