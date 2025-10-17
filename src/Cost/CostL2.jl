
struct CostL2{I,D<:Union{AbstractArray,Number}} <: AbstractCost{I}
	inputspace::I
	data::D
	#W::P{T}    # precision matrix   

	CostL2(inputspace::I, data::D) where {I<:CoordinateSpace,D<:Union{AbstractArray,Number}} =  new{I,D}(inputspace,data)
end



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


function CostL2(sz::NTuple{N,Int},data::T) where {N,T<:Number}  
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



struct scaledCostL2{I,D<:Union{AbstractArray,Number},P,N} <: AbstractCost{I}
	inputspace::I
	wdata::D
	W::P    # precision matrix   
	dims::N
	scaledCostL2(inputspace::I, wdata::D,W::P,dims::N) where {I<:CoordinateSpace,D<:Union{AbstractArray,Number},P,N} =  new{I,D,P,N}(inputspace,wdata,W,dims)
end

 

function scaledCostL2(data::D; W=LinOpIdentity(size(data)),dims=Tuple( 1:N)) where {N,T,D<:AbstractArray{T,N}}  
	if (W isa AbstractArray{T,N}) 
		W = LinOpDiag(W)
	end
	(W isa AbstractLinOp) || throw(SimpleAlgebraFailure("W must a LinOp or an array"))

	sz = size(data)
	inputspace = CoordinateSpace(T,sz)
	wdata = W*data
	return scaledCostL2(inputspace, wdata,W,dims) 
end

function apply_(A::scaledCostL2, v) 
	wv = A.W*v
	α = sum(v.*A.wdata,dims=A.dims) ./ sum(wv.*v,dims=A.dims)
	return sum(abs2,α .* wv .- A.wdata)/2
end

function apply_jacobian_(A::scaledCostL2, v, x)
	wv = A.W*v
	α = sum(v.*A.wdata,dims=A.dims) ./ sum(wv.*v,dims=A.dims)
	return  α .* x .* (α .* wv .- A.wdata)
end
 
function getalpha(A::scaledCostL2, v) 
	wv = A.W*v
	return sum(v.*A.wdata,dims=A.dims) ./ sum(wv.*v,dims=A.dims)
end

function ChainRulesCore.rrule( ::typeof(apply_),A::scaledCostL2, v)
	wv = A.W*v
	α = sum(v.*A.wdata,dims=A.dims) ./ sum(wv.*v,dims=A.dims)
	r = α .* wv .- A.wdata
    costL2_pullback(Δy) = (NoTangent(),NoTangent(), α .* r .* Δy)
    return  sum(abs2,r)/2, costL2_pullback
end
