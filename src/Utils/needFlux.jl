
# Should probably be replaced by Functors.jl . However not sure that adding @functors for all object is easier to write



#= function Flux.gpu(M::LinOpDFT{I,O,TI,TO,F,B}) where {I,O,TI,TO,F,B}
    temp = Array{ComplexF32}(undef, size(I)) |> gpu
	T = ComplexF32
    forward = plan_fft(temp)
    backward = plan_fft(temp;)

    # Build operator.
	_I = CoordinateSpace{forward.sz}
	_O = CoordinateSpace{forward.osz}
    _F = typeof(forward)
    _B = typeof(backward)
    return LinOpDFT{_I,_O,T,T,_F,_B}(forward, backward)
end
 =#  

#= function Flux._paramtype(::Type{T}, m::LinOpDFT) where {T<:fftwNumber}
  convert(T, m)
end
 =#
#= 
function Adapt.adapt_storage(::Type{T}, x::LinOpDFT) where {T<:fftwNumber}
    dims = inputsize(x)
    planning = planning = check_flags(FFTW.MEASURE)
    timelimit = FFTW.NO_TIMELIMIT
    # Compute the plans with suitable FFTW flags.  For maximum efficiency, the
    # transforms are always applied in-place and thus cannot preserve their
    # inputs.
    
    if T<: fftwReal
        forward = plan_rfft(Array{T}(undef, dims);
                        flags = (planning | FFTW.PRESERVE_INPUT),
                        timelimit = timelimit)

        backward = plan_brfft(Array{Complex{T}}(undef, forward.osz), dims[1];
                          flags = (planning | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)
    else
        temp = Array{T}(undef, dims)
        forward = plan_fft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                        timelimit = timelimit)
        backward = plan_bfft!(temp; flags = (planning | FFTW.DESTROY_INPUT),
                          timelimit = timelimit)
    end


    # Build operator.

	inputspace = CoordinateSpace(T,forward.sz)
	outputspace = CoordinateSpace(T,forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)
end =#
#import Flux: FluxEltypeAdaptor
Adapt.adapt_storage(::Flux.FluxEltypeAdaptor{T}, x::AbstractMap)  where T = Adapt.adapt_storage(T, x)
