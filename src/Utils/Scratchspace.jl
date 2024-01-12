struct Scratchspace{A}  
	scratch::A
	function Scratchspace(scratch::M) where {M<:DenseVector}
		new{typeof(scratch)}(scratch)
	end
end

function Scratchspace(::Type{M}) where {T, M<:DenseArray{T}}
	scratch = Base.typename(M).wrapper{T,1}(undef,1)
	Scratchspace(scratch)
end
function Scratchspace(::Type{M},::T) where {T, M<:DenseArray}
	scratch = Base.typename(M).wrapper{T,1}(undef,1)
	Scratchspace(scratch)
end

#= function newarray!(ssp::Scratchspace{A},csp::CoordinateSpace{T,N}) where {T,N,A<:AbstractVector{T}}
    len = length(csp)
	if length(ssp.scratch) < len
		resize!(ssp.scratch,len)
	end
	return reshape(view(ssp.scratch,1:len),size(csp))
end =#

function newarray!(ssp::Scratchspace{A},csp::CoordinateSpace{T,N}) where {T,T2,N,A<:DenseVector{T2}}
	T3 =  isconcretetype(T) ? T : T2
    bitlen =  length(csp) * sizeof(T3)
	if sizeof(ssp.scratch) < bitlen
		len = div( bitlen , sizeof(T2), RoundUp)
		resize!(ssp.scratch,len)
	end
	#return reshape(view(reinterpret(T,ssp.scratch),1:length(csp)),size(csp))
	return unsafe_wrap(Base.typename(A).wrapper,pointer(reinterpret(T3,ssp.scratch)),size(csp))
end

clear!(ssp::Scratchspace) = resize!(ssp.scratch,1)

Base.resize!(ssp::Scratchspace, nl) = resize!(ssp.scratch,nl)

Adapt.adapt_structure(to,ssp::Scratchspace) = Scratchspace(adapt(to,ssp.scratch))