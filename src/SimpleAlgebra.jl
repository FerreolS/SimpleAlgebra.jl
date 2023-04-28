module SimpleAlgebra
import Flux: FluxCUDAAdaptor, gpu

using ChainRulesCore
using Flux
struct SimpleAlgebraFailure <: Exception
    msg::String
end

showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)

include("Domain.jl")
include("Map.jl")
include("LinOp/LinOp.jl")
include("Cost/Cost.jl")


function Flux.gpu(::Flux.FluxCUDAAdaptor,M::AbstractMap) 
	n = nfields(M)
	n==0 && return M
	Mfields = (getfield(M,i) for i ∈ 1:n)
	Mfields = map(gpu,Mfields)
	typeof(M)(Mfields...)
end
#Adapt.adapt(Flux.FluxCUDAAdaptor(),L1)

end # module SimpleAlgebra
