
struct CostL2{I,D<:Union{AbstractArray,Number}} <:AbstractCost{I}
	data::D
	#W::P{T}    # precision matrix   
end

function CostL2(sz::NTuple,data::T1) where {T1<:Number}  
	return CostL2{sz,T1}(data)
end

function CostL2(::Type{T}, data::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	data = convert.(T, data)
	return CostL2(data)
end

function CostL2(data::D) where {T<:Number,D<:AbstractArray{T}}  
	sz = size(data)
	return CostL2(sz,data)
end
function CostL2(sz::NTuple,data::D) where {T<:Number,D<:AbstractArray{T}}  
	@assert sz==size(data) "the data should be of size $sz"
	return CostL2{sz,D}(data)
end


function apply(A::CostL2{I,D}, v) where {I,D}
	return 0.5 * sum(abs2,v .- A.data)
end


function ChainRulesCore.rrule( ::typeof(apply),A::CostL2, v)
	r = v .- A.data
    ∂Y(Δy) = (NoTangent(),NoTangent(), r .* Δy)
    return 0.5 * sum(abs2,r), ∂Y
end