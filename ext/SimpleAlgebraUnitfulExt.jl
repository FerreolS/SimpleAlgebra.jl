module SimpleAlgebraUnitfulExt
using Unitful, Adapt
	#FIXME: Type piracy
#	Adapt.adapt_storage(::Type{A}, xs::AT) where {T,A<:AbstractArray{T},Q<:Unitful.Quantity,AT<:AbstractArray{Q}} = adapt(Base.typename(typeof(A)).wrapper,T.(xs))
	Adapt.adapt_storage(::Type{T}, x::AT)  where {T<:Number,Q<:Unitful.Quantity,AT<:AbstractArray{Q}} =  T.(x)

end