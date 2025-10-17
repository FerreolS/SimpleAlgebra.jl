struct CostHyperbolic{I, L <: MapReduceSum, Q <: MapReduceSum, E <: Union{AbstractMap, AbstractArray, Number}, P <: AbstractMap} <: AbstractCost{I}
    inputspace::I
    innerSum::L
    outerSum::Q
    ϵ::E
    operator::P
    CostHyperbolic(inputspace::I, innerSum::L, outerSum::Q, ϵ::E, operator::P) where
    {I, L <: MapReduceSum, Q <: MapReduceSum, E <: Union{AbstractMap, AbstractArray, Number}, P <: AbstractMap} =
        new{I, L, Q, E, P}(inputspace, innerSum, outerSum, ϵ, operator)
end


function CostHyperbolic(::Type{T}, sz) where {T}
    inputspace = CoordinateSpace(T, sz)
    operator = LinOpGrad(sz)
    innerSum = MapReduceSum(outputsize(operator), abs2, ndims(outputspace(operator)))
    outerSum = MapReduceSum(outputsize(innerSum), sqrt)
    ϵ = MapReduceSum(sz, abs2)
    return CostHyperbolic(inputspace, innerSum, outerSum, ϵ, operator)
end

CostHyperbolic(sz::NTuple) = CostHyperbolic(Number, sz)

apply_(A::CostHyperbolic, x) = A.outerSum * (A.innerSum * A.operator + A.ϵ) * x
