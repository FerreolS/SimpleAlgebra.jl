module SimpleAlgebraLinearAlgebraExt
	using SimpleAlgebra, LinearAlgebra
	Base.:+(S::LinearAlgebra.UniformScaling,A::AbstractMap)  = S.λ + A
	Base.:+(A::AbstractMap,S::LinearAlgebra.UniformScaling)  = A +  S.λ
	Base.:-(S::LinearAlgebra.UniformScaling,A::AbstractMap)  = S.λ - A
	Base.:-(A::AbstractMap,S::LinearAlgebra.UniformScaling)  = A -  S.λ
	Base.:*(S::LinearAlgebra.UniformScaling,A::AbstractMap)  = S.λ * A
	Base.:*(A::AbstractMap,S::LinearAlgebra.UniformScaling)  = A *  S.λ
end