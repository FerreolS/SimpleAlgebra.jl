module SimpleAlgebraFluxExt
using SimpleAlgebra,Flux, Adapt,CUDA
	Adapt.adapt_storage(to::Flux.FluxCUDAAdaptor, x::AbstractMap) = adapt(CuArray{Float32}, x)
	Adapt.adapt_storage(to::Flux.FluxCPUAdaptor, x::AbstractMap) = adapt(Array, x)
end