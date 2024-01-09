module SimpleAlgebraFluxExt
using Flux, Adapt, SimpleAlgebra


Adapt.adapt_storage(::Flux.FluxEltypeAdaptor{T}, x::AbstractMap)  where T = Adapt.adapt_storage(AbstractArray{T}, x)
end