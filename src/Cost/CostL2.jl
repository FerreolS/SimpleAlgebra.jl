
struct CostL2{I,D<:Union{AbstractArray,Number}} <: AbstractCost{I}
	data::D
	#W::P{T}    # precision matrix   
end

function CostL2(sz::NTuple,data::T) where {T<:Number}  
	return CostL2{CoordinateSpace{sz},T}(data)
end

function CostL2(::Type{T}, data::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	data = convert.(T, data)
	return CostL2(data)
end

function CostL2(data::D) where {D<:AbstractArray}  
	sz = size(data)
	return CostL2(sz,data)
end

function CostL2(sz::NTuple,data::D) where {D<:AbstractArray}  
	@assert sz==size(data) "the data should be of size $sz"
	return CostL2{CoordinateSpace{sz},D}(data)
end


function apply(A::CostL2{I,D}, v) where {I,D}
	return 0.5 * sum(abs2,v .- A.data)
end


function ChainRulesCore.rrule( ::typeof(apply),A::CostL2, v)
	r = v .- A.data
    ∂Y(Δy) = (NoTangent(),NoTangent(), r .* Δy)
    return 0.5 * sum(abs2,r), ∂Y
end