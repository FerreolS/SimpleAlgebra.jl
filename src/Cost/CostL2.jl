
struct CostL2{T,I,D<:Union{AbstractArray{T},T}} <:AbstractCost{T,I}
	data::D
	#W::P{T}    # precision matrix   
end

CostL2(sz::NTuple,data::T) where {T<:Number}  =  CostL2(T,sz,data)
function CostL2(::Type{T}, sz::NTuple,data::T1) where {T<:Number,T1<:Number}  
	return CostL2{T,sz,T}(data)
end

function CostL2(::Type{T}, data::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	data = convert.(T, data)
	return CostL2(T,data)
end
function CostL2(::Type{T}, data::D) where {T<:Number,D<:AbstractArray{T}}  
	sz = size(data)
	return CostL2(T,sz,data)
end
function CostL2(::Type{T}, sz::NTuple,data::D) where {T<:Number,D<:AbstractArray{T}}  
	@assert sz==size(data) "the data should be of size $sz"
	return CostL2{T,sz,D}(data)
end
CostL2(data::D) where {T<:Number,D<:AbstractArray{T}}  =  CostL2(T, data)

CostL2{T}(obj::CostL2{T2,I,D}) where {T,T2,I,D} = CostL2(T,I,convert.(T,obj.data))

function apply(A::CostL2{T,I,D}, v) where {T,I,D}
	return 0.5 * sum(abs2,v .- A.data)
end


function ChainRulesCore.rrule( ::typeof(apply),A::CostL2, v)
	r = v .- A.data
    ∂Y(Δy) = (NoTangent(),NoTangent(), r .* Δy)
    return 0.5 * sum(abs2,r), ∂Y
end