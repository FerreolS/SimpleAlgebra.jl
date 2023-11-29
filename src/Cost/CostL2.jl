
struct CostL2{I,D<:Union{AbstractArray,Number}} <: AbstractCost{I}
	inputspace::I
	data::D
	#W::P{T}    # precision matrix   

	CostL2(inputspace::I, data::D) where {I<:CoordinateSpace,D<:Union{AbstractArray,Number}} =  new{I,D}(inputspace,data)
end

@functor CostL2


function CostL2(::Type{T}, data::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	data = convert.(T, data)
	return CostL2(data)
end

function CostL2(data::D) where {T,D<:AbstractArray{T}}  
	sz = size(data)
	inputspace = CoordinateSpace(T,sz)
	return CostL2(inputspace, data)
end


function CostL2(::Type{TI}, sz::NTuple{N,Int},data::T) where {TI,N,T<:Number}  
	inputspace = CoordinateSpace(TI,sz)
	return CostL2(inputspace, data)
end


function CostL2(sz::NTuple{N,Int},data::T) where {TI,N,T<:Number}  
	inputspace = CoordinateSpace(T,sz)
	return CostL2(inputspace, data)
end

function apply_(A::CostL2{I,D}, v) where {I,D}
	return sum(abs2,v .- A.data)/2
end

function apply_jacobian_(A::CostL2, v, x)
	return @. x * (v - A.data)
end
 

 # FIXME 
function ChainRulesCore.rrule( ::typeof(apply_),A::CostL2, v)
	r = v .- A.data
    costL2_pullback(Δy) = (NoTangent(),NoTangent(), r .* Δy)
    return  sum(abs2,r)/2, costL2_pullback
end