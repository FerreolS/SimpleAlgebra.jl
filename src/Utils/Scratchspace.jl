struct Scratchspace{A}  
	scratch::A
	function Scratchspace(scratch::M) where {M<:AbstractVector}
		new{typeof(scratch)}(scratch)
	end
end

function Scratchspace(::Type{M}) where {T, M<:AbstractArray{T}}
	scratch = Base.typename(M).wrapper{T,1}(undef,1)
	Scratchspace(scratch)
end

function newarray(csp::CoordinateSpace{T,N}, ssp::Scratchspace{A}) where {T,N,A<:AbstractVector{T}}
    len = length(csp)
	if length(ssp.scratch) < len
		resize!(ssp.scratch,len)
	end
	return reshape(view(ssp.scratch,1:len),size(csp))
end

function newarray(csp::CoordinateSpace{T,N}, ssp::Scratchspace{A}) where {T,T2,N,A<:AbstractVector{T2}}
    bitlen = length(csp) * sizeof(eltype(csp))
	len = div( bitlen , sizeof(T2), RoundUp)
	if sizeof(ssp.scratch) < bitlen
		resize!(ssp.scratch,len)
	end
	return reshape(view(reinterpret(T,ssp.scratch),1:length(csp)),size(csp))
end

clear!(ssp::Scratchspace) = resize!(ssp.scratch,1)

Base.resize!(ssp::Scratchspace, nl) = resize!(ssp.scratch,nl)

Adapt.adapt_structure(to,ssp::Scratchspace) = Scratchspace(adapt(to,ssp.scratch))