
# Should probably be replaced by Functors.jl . However not sure that adding @functors for all object is easier to write
function Flux.gpu(M::AbstractMap) 
	n = nfields(M)
	n==0 && return M
	Mfields = (getfield(M,i) for i ∈ 1:n)
	Mfields = map(gpu,Mfields)
	typeof(M)(Mfields...)
end


function Flux.gpu(M::LinOpDFT{I,O,TI,TO,F,B}) where {I,O,TI,TO,F,B}
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