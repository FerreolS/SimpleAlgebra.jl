module SimpleAlgebra
using ChainRulesCore

struct SimpleAlgebraFailure <: Exception
    msg::String
end

showerror(io::IO, err::SimpleAlgebraFailure) =
    print(io, err.msg)

include("Domain.jl")
include("Map.jl")
include("LinOp/LinOp.jl")
include("Cost/Cost.jl")
end # module SimpleAlgebra
