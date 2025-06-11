module jlSAT

# Julia formulation of CTDS-SAT, with higher order correlations
# Written by Áron Vízkeleti
# Last modified 2025-06-11
# 
# Copyright (c) Áron Vízkeleti
# All rights reserved.
#
# This code is protected under copyright law.
# Unauthorized copying, modification, or distribution is prohibited.

using Random, LinearAlgebra
using DifferentialEquations
import Distributions: Normal

function load_cnf(file_name::String)::Matrix{Int8}
    c = Nothing
    open(file_name) do file
        for (idx, line) in enumerate(eachline(file))
            if idx == 1
                N = parse(Int32, split(line, " ")[3])
                M = parse(Int32, split(line, " ")[4])
                c = zeros(Int8, M,N)
            else
                variables = split(line, " ")
                for var_str in variables
                    var = parse(Int32, var_str)
                    if var != 0
                        if var > 0
                            c[idx-1, var] = 1
                        elseif var < 0
                            c[idx-1, -var] = -1
                        end
                    end
                end
            end
        end
    end
    return c
end

function satisfied(spin_config, c::Matrix{Int8})::Bool
    function check_clause(row, state)::Bool
        for (index,elem) in enumerate(row)
            if elem == state[index]
                return true
            end
        end
        return false
    end

    incorrect_flag = false
    for clause in eachrow(c)
        if check_clause(clause, spin_config) != true
            incorrect_flag = true
            break
        end
    end

    if incorrect_flag
        return false
    end
    return true
end


function precompute_K!(K::Array{Float64}, c::Matrix{Int8}, s::Vector{Float64}, factors::Vector{Float64})::Nothing
    M, N = size(c)
    for m ∈ 1:M
        @inbounds K[m] = prod( @inbounds (1 .- (c[m,:] .* s) )) * factors[m]
    end
end

function OLDprecompute_κ!(κ::Array{Float64}, c::Matrix{Int8}, s::Vector{Float64}, factors::Vector{Float64})::Nothing
    M, N = size(c)
    temp = similar(Array{Float64}, axes(c))
    for m ∈ 1:M
        @inbounds temp[m, :] .= 1 .- (c[m, :] .* s)
        for i ∈ 1:N
            @inbounds κ[m, i] = prod(temp[m, [1:i-1; i+1:N]]) * factors[m]
        end
    end
end

function product_except_self!(result::Vector{Float64}, K::Vector{Float64})::Nothing
    N = length(K)
    prefix = 1.0
    @inbounds for i in 1:N
        result[i] = prefix
        prefix *= K[i]
    end
    suffix = 1.0
    @inbounds for i in N:-1:1
        result[i] *= suffix
        suffix *= K[i]
    end
end

function precompute_κ!(κ::Array{Float64}, c::Matrix{Int8}, s::Vector{Float64}, factors::Vector{Float64})::Nothing
    M, N = size(c)
    temp = Vector{Float64}(undef, N)
    #@inbounds temp .= 1 .- c[m, :] .* s
    row_products = Vector{Float64}(undef, N)

    for m in 1:M
        @inbounds for i in 1:N
            temp[i] = 1 - c[m, i] * s[i]
        end

        product_except_self!(row_products, temp)

        @inbounds for i in 1:N
            κ[m, i] = row_products[i] * factors[m]
        end
    end
end

#function NEWOLD2precompute_κ!(κ::Array{Float64}, c::Matrix{Int8}, s::Vector{Float64}, factors::Vector{Float64})::Nothing
#    M, N = size(c)
#    temp = Vector{Float64}(undef, N)
#    @inbounds temp .= 1 .- c[m, :] .* s
#    row_products = Vector{Float64}(undef, N)
#
#    for m in 1:M
#        @inbounds for i in 1:N
#            temp[i] = 1 - c[m, i] * s[i]
#        end
#
#        product_except_self!(row_products, temp)
#
#        @inbounds for i in 1:N
#            κ[m, i] = row_products[i] * factors[m]
#        end
#    end
#end
#
#function NEWprecompute_κ!(
#    κ::Matrix{Float64},
#    c::Matrix{Int8},
#    s::Vector{Float64},
#    factors::Vector{Float64}
#)::Nothing
#    M, N = size(c)
#    temp = Vector{Float64}(undef, N)
#    row_products = Vector{Float64}(undef, N)
#
#    @inbounds for m in 1:M
#        @views temp .= 1 .- c[m, :] .* s         # efficient, fused, no bounds check
#        product_except_self!(row_products, temp)
#        @inbounds κ[m, :] .= row_products .* factors[m]  # elementwise multiply row
#    end
#end



function hgCTDS_lor_rule_old!(du::Vector{Float64}, u::Vector{Float64}, p::Tuple, t::Float64)::Nothing
    c = p[1]
    HG = p[2]
    λ = p[3]
    norm_factors = p[4]
    #K = p[5] # Pass pre-allocated K vector by reference
    #κ = p[6]

    M, N = size(c)
    s = u[1:N]
    Γ = u[N+1:end]

    du[1:N] .= zeros(N) # Is this neccessary? 

    if p[end] ≠ nothing
        start_time = time()
    end
    
    K = Array{Float64}(undef, M)
    κ = Array{Float64}(undef, M, N)

    #Update K vector and κ matrix in place
    precompute_K!(K, c, s, norm_factors)  
    precompute_κ!(κ, c, s, norm_factors)

    for (μ, hyper_edge) ∈ enumerate(HG)
        χ_μ = 1.0

        for m ∈ hyper_edge
            χ_μ *= K[m] * K[m]
        end
        
        if λ == 0.0
            du[N+μ] = Γ[μ]* χ_μ
        else
            du[N+μ] = Γ[μ]*( χ_μ - λ * log(Γ[μ]) )
        end

        for i ∈ 1:N
            inner_summ = 0.0
            for m ∈ hyper_edge
                if c[m, i] ≠ 0
                    inner_summ += c[m, i] * κ[m, i] * K[m]
                end
            end
            du[i] += Γ[μ] * inner_summ
        end

    end

    if p[end] ≠ nothing
        elapsed_time = time() - start_time
        push!(p[end-1][], elapsed_time)  # Update the elapsed time vector
        push!(p[end][], t)               # Update analog time
    end
    return nothing
end

function precalculate_land_χ_μn!(prod_vec, K, μ)
    for (idx, n) ∈ enumerate(μ)
        product = 1.0
        for m ∈ μ
            if m ≠ n
                product *= 1 - K[m]^2
            end
        end
        prod_vec[idx] = product
    end
end

function precompute_land_χ!(χ, K, HG)
    for (μ, hyper_edge) ∈ enumerate(HG)
        χ_μ = 1.0
        for m ∈ hyper_edge
            χ_μ *= 1 - K[m]^2
        end
        χ[μ] = 1 - χ_μ
    end
end

function precalculate_lor_χ_μn_old!(prod_vec, K, μ)
    for (idx, n) ∈ enumerate(μ)
        product = 1.0
        for m ∈ μ
            if m ≠ n
                product *= K[m]^2
            end
        end
        prod_vec[idx] = product
    end
end

function precalculate_lor_χ_μn!(prod_vec::Vector{Float64}, K::Vector{Float64}, μ::Vector{Int})::Nothing
    L = length(μ)

    #Handling small hyper edges separately for speed (woah!)
    if L == 1
        prod_vec[1] = 1.0
        return
    elseif L == 2
        @inbounds prod_vec[1] = K[μ[2]]^2
        @inbounds prod_vec[2] = K[μ[1]]^2
        return
    end

    squared_Kμ = Vector{Float64}(undef, L)
    prefix = Vector{Float64}(undef, L)
    suffix = Vector{Float64}(undef, L)

    @inbounds for i in 1:L
        squared_Kμ[i] = K[μ[i]]^2
    end

    prefix[1] = 1.0
    @inbounds for i in 2:L
        prefix[i] = prefix[i-1] * squared_Kμ[i-1]
    end

    suffix[L] = 1.0
    @inbounds for i in L-1:-1:1
        suffix[i] = suffix[i+1] * squared_Kμ[i+1]
    end

    @inbounds for i in 1:L
        prod_vec[i] = prefix[i] * suffix[i]
    end
end

#Usually not needed
function precompute_lor_χ!(χ, K, HG)
    for (μ, hyper_edge) ∈ enumerate(HG)
        χ[μ] = 1.0
        for m ∈ hyper_edge
            χ[μ] *= K[m]^2
        end
    end
end

function hgCTDS_lor_rule!(du::Vector{Float64}, u::Vector{Float64}, p::Tuple, t::Float64)::Nothing
    c = p[1]
    HG = p[2]
    λ = p[3]
    norm_factors = p[4]
    K = p[5] # Pass pre-allocated K vector by reference
    κ = p[6]

    M, N = size(c)
    s = u[1:N]
    Γ = u[N+1:end]

    du[1:N] .= zeros(N) # Is this neccessary? 

    if p[end] ≠ nothing
        start_time = time()
    end
    
    #Update K vector and κ matrix in place
    precompute_K!(K, c, s, norm_factors)  
    precompute_κ!(κ, c, s, norm_factors)

    for (μ, hyper_edge) ∈ enumerate(HG)
        χ_μ = 1.0 # No need to precalculate & allocate the full vector

        
        χ_μm = Array{Float64}(undef, length(hyper_edge))
        precalculate_lor_χ_μn!(χ_μm, K, hyper_edge)

        for m ∈ hyper_edge
            χ_μ *= K[m]
            #χ_μ *= K[m]^2
        end
        
        if λ == 0.0
            du[N+μ] = Γ[μ]* χ_μ #Actually sqrt(χ_μ)? No, this should be fine
        else
            du[N+μ] = Γ[μ]*( χ_μ - λ * log(Γ[μ]) )
        end

        for i ∈ 1:N
            inner_summ = 0.0
            for (hyperedge_idx, m) ∈ enumerate(hyper_edge)
                if c[m, i] ≠ 0
                    inner_summ += 2 * c[m, i] * κ[m, i] * K[m] * χ_μm[hyperedge_idx]^2
                end
            end
            du[i] += Γ[μ] * inner_summ
        end

    end

    if p[end] ≠ nothing
        elapsed_time = time() - start_time
        push!(p[end-1][], elapsed_time)  # Update the elapsed time vector
        push!(p[end][], t)               # Update analog time
    end
    return nothing
end

#A wrapper function to calculate a trajectory 
function get_basin(initial_s::Vector{Float64}, params::Dict{String, Any}, DEBUG_FLAG::Bool = false)::Dict{String, Any}
    c = params["c"]
    HG = params["HG"]
    t_max = params["t_max"]
    idx_1 = params["idx_1"]
    idx_2 = params["idx_2"]
    k_factors = params["k_factors"]

    condition(u, t, integrator) = satisfied([s>0 ? 1 : -1 for s in u], c)
    affect!(integrator) = terminate!(integrator)
    CTDS_cb = DiscreteCallback(condition, affect!)
    function kBALL_condition(u, t, integrator)
        c = integrator.p[1]
    
        norm_factors = integrator.p[4]
        K = integrator.p[5]
        M, N = size(c)
        s = u[1:N]
        
        jlSAT.precompute_K!(K, c, s, norm_factors)  
    
        return norm(K) < 0.001
    end

    BALL_cb = DiscreteCallback(kBALL_condition, affect!)

    M, N = size(c)

    L = length(HG)

    u0 = Array{Float64}(undef, L + N)
    u0[1:N] .= initial_s
    u0[N+1:end] .= ones(L);

    
    K = Array{Float64}(undef, M)
    κ = Array{Float64}(undef, M, N)
    
    p = (c, HG, 0.0, k_factors, K, κ, nothing, nothing)
    tspan = (0.0,t_max);

    prob = ODEProblem(hgCTDS_lor_rule!, u0, tspan, p);
    sol = solve(prob, callback = BALL_cb);

    result_dict = Dict(
        "max_time" => sol.t[end],
        "final_state" => sol.u[end],
        "satisfied" => satisfied([s>0 ? 1 : -1 for s in sol.u[end][1:N]], c),
        "initial_state" => initial_s
    )

    if DEBUG_FLAG
        pix_1 = initial_s[idx_1]
        pix_2 = initial_s[idx_2]
        println("Pixel at $pix_1 $pix_2")
    end

    return result_dict
end

#A wrapper function to calculate a trajectory 
function get_sampled_trajectory(initial_s::Vector{Float64}, params::Dict{String, Any}, DEBUG_FLAG::Bool = false)::Dict{String, Any}
    c = params["c"]
    HG = params["HG"]
    t_max = params["t_max"]
    k_factors = params["k_factors"]
    time_points = params["time_points"]

    condition(u, t, integrator) = satisfied([s>0 ? 1 : -1 for s in u], c)
    affect!(integrator) = terminate!(integrator)
    #CTDS_cb = DiscreteCallback(condition, affect!)
    function kBALL_condition(u, t, integrator)
        c = integrator.p[1]
    
        norm_factors = integrator.p[4]
        K = integrator.p[5]
        M, N = size(c)
        s = u[1:N]
        
        jlSAT.precompute_K!(K, c, s, norm_factors)  
    
        return norm(K) < 0.001
    end

    BALL_cb = DiscreteCallback(kBALL_condition, affect!)

    M, N = size(c)
    L = length(HG)

    t_sol = 0.0
    trial_counter = 0
    exit_condition = false
    result_dict = Dict()

    while trial_counter < params["nbr_trial"] && !exit_condition

        u0 = Array{Float64}(undef, L + N)
        u0[1:N] .= initial_s
        u0[N+1:end] .= ones(L);

        
        K = Array{Float64}(undef, M)
        κ = Array{Float64}(undef, M, N)
        
        p = (c, HG, 0.0, k_factors, K, κ, nothing, nothing)
        tspan = (0.0,t_max);

        prob = ODEProblem(hgCTDS_lor_rule!, u0, tspan, p);
        sol = nothing  # Initialize sol to avoid undefined variable issues
        try
            sol = solve(prob, callback = BALL_cb)
        catch e
            println("ODE solver failed: ", e)
            return Dict("status" => "solver_error", "initial_state" => initial_s)
        end

        if sol.t[end] >= t_sol
            t_sol = sol.t[end]
            result_dict = Dict(
                "max_time" => sol.t[end],
                "final_state" => sol.u[end],
                "satisfied" => satisfied([s>0 ? 1 : -1 for s in sol.u[end][1:N]], c),
                "initial_state" => initial_s,
                "time_series" => [sol(t) for t in time_points]
            )
        end

        if t_sol >= params["t_min"] || trial_counter >= params["nbr_trial"]
            exit_condition = true
        end

        trial_counter += 1
    end


    if DEBUG_FLAG
        println("A run finished")
    end

    return result_dict
end

function separation_simulation(initial_s::Vector{Float64}, params::Dict{String, Any}, DEBUG_FLAG::Bool = false)::Dict{String, Any}
    c = params["c"]
    HG = params["HG"]
    t_max = params["t_max"]
    k_factors = params["k_factors"]
    
    M, N = size(c)
    L = length(HG)
    u0 = Array{Float64}(undef, L + N)
    u0[1:N] .= initial_s
    u0[N+1:end] .= ones(L);
    K = Array{Float64}(undef, M)
    κ = Array{Float64}(undef, M, N)
    p = (c, HG, 0.0, k_factors, K, κ, nothing, nothing)
    tspan = (0.0,t_max);
    
    affect!(integrator) = terminate!(integrator)
    function kBALL_condition(u, t, integrator)
        c = integrator.p[1]
    
        norm_factors = integrator.p[4]
        K = integrator.p[5]
        M, N = size(c)
        s = u[1:N]
        
        jlSAT.precompute_K!(K, c, s, norm_factors)  
    
        return norm(K) < 0.001
    end
    BALL_cb = DiscreteCallback(kBALL_condition, affect!)

    prob = ODEProblem(hgCTDS_lor_rule!, u0, tspan, p);
    sol = solve(prob, callback = BALL_cb);

    result_dict = Dict(
        "time" => sol.t,
        "trajectory" => [(xₜ[1],xₜ[2])  for xₜ ∈ sol.u]
    )

    return result_dict
end

#A wrapper function to calculate a trajectory 
function get_basin_evolution(initial_s::Vector{Float64}, params::Dict{String, Any}, DEBUG_FLAG::Bool = false)::Dict{String, Any}
    c = params["c"]
    HG = params["HG"]
    t_max = params["t_max"]
    idx_1 = params["idx_1"]
    idx_2 = params["idx_2"]
    k_factors = params["k_factors"]

    affect!(integrator) = terminate!(integrator)
    function kBALL_condition(u, t, integrator)
        c = integrator.p[1]
    
        norm_factors = integrator.p[4]
        K = integrator.p[5]
        M, N = size(c)
        s = u[1:N]
        
        jlSAT.precompute_K!(K, c, s, norm_factors)  
    
        return norm(K) < 0.001
    end
    
    BALL_cb = DiscreteCallback(kBALL_condition, affect!)

    M, N = size(c)

    L = length(HG)

    u0 = Array{Float64}(undef, L + N)
    u0[1:N] .= initial_s
    u0[N+1:end] .= ones(L);

    K = Array{Float64}(undef, M)
    κ = Array{Float64}(undef, M, N)
    
    p = (c, HG, 0.0, k_factors, K, κ, nothing, nothing)
    tspan = (0.0,t_max);

    prob = ODEProblem(hgCTDS_lor_rule!, u0, tspan, p);
    sol = solve(prob, callback = BALL_cb);

    result_dict = Dict(
        "time" => sol.t,
        "trajectory" => [(xₜ[idx_1],xₜ[idx_2])  for xₜ ∈ sol.u]
    )

    if DEBUG_FLAG
        pix_1 = initial_s[idx_1]
        pix_2 = initial_s[idx_2]
        println("Pixel at $pix_1 $pix_2")
    end

    return result_dict
end

function CTDS_rule!(du, u, p, t)
    start_time = time()

    c = p[1]
    norm_factors = p[4]
    M, N = size(c)


    s = u[1:N]
    a = u[N+1:end]
    for i in 1:N 
        du[i] = sum([2 * a[m]*c[m,i]*(norm_factors[m]*prod([ j!=i ? (1-c[m, j]*s[j]) : 1 for j in 1:length(s)]))^2*(1-c[m, i]*s[i]) for m in 1:M])
    end
    for m in 1:M
        du[N+m] = a[m]*norm_factors[m]*prod([ (1-c[m, j]*s[j]) for j in 1:N])
    end

    elapsed_time = time() - start_time
    push!(p[end-1][], elapsed_time)  # Update the elapsed time vector
    push!(p[end][], t)
    return nothing
end

function CTDSl_rule!(du, u, p, t)
    c = p[1]
    #HG = p[2]
    λ = p[3]
    norm_factors = p[4]

    M, N = size(c)
    s = u[1:N]
    a = u[N+1:end]
    for i in 1:N 
        du[i] = sum([a[m]*c[m,i]*(norm_factors[m]*prod([ j!=i ? (1-c[m, j]*s[j]) : 1 for j in 1:N]))^2*(1-c[m, i]*s[i]) for m in 1:M])
    end
    for m in 1:M
        du[N+m] = a[m]*(-λ*log(a[m]) + norm_factors[m]*prod([ (1-c[m, j]*s[j]) for j in 1:N]))
    end
    return nothing
end

function reparam_base_CTDS_rule!(du, u, p, t)
    if p[end] ≠ nothing
        start_time = time()
    end
    

    c = p[1]
    norm_factors = p[4]
    M, N = size(c)


    s = u[1:N]
    y = u[N+1:end]
    a = exp.( t .* y ) # VERY UNSTABLE!!!
    for i in 1:N 
        du[i] = sum([a[m]*c[m,i]*(norm_factors[m]*prod([ j!=i ? (1-c[m, j]*s[j]) : 1 for j in 1:length(s)]))^2*(1-c[m, i]*s[i]) for m in 1:M])
    end
    for m in 1:M
        k_m = norm_factors[m]*prod([ (1-c[m, j]*s[j]) for j in 1:N])
        du[N+m] = ( k_m - y[m] )/t
    end

    if p[end] ≠ nothing
        elapsed_time = time() - start_time
        push!(p[end-1][], elapsed_time)  # Update the elapsed time vector
        push!(p[end][], t)               # Update analog time
    end

    return nothing
end

function reparam_base_CTDS_BACKWARDrule!(du, u, p, t)
    reparam_base_CTDS_rule!(du, u, p, -t)
    du .= -du
end

function dsdt(u, t::Real, p::Tuple)
    c = p[1]
    norm_factors = p[4]
    M, N = size(c)
    s = u[1:N]
    y = u[N+1:end]
    a = exp.( t .* y )
    du = similar(s)

    for i in 1:N 
        du[i] = sum([a[m]*c[m,i]*(norm_factors[m]*prod([ j!=i ? (1-c[m, j]*s[j]) : 1 for j in 1:length(s)]))^2*(1-c[m, i]*s[i]) for m in 1:M])
    end

    return du
end

function hgdsdt(u::Vector{Float64}, p::Tuple)::Vector{Float64}
    c = p[1]
    HG = p[2]
    λ = p[3]
    norm_factors = p[4]

    M, N = size(c)
    s = u[1:N]
    Γ = u[N+1:end]

    du = similar(s)
    du[1:N] .= zeros(N) # Is this neccessary? 
    
    K = Array{Float64}(undef, M)
    precompute_K!(K, c, s, norm_factors)

    κ = Array{Float64}(undef, M, N)
    precompute_κ!(κ, c, s, norm_factors)

    for (μ, hyper_edge) ∈ enumerate(HG)
        χ_μ = 1.0

        χ_μm = Array{Float64}(undef, length(hyper_edge))
        precalculate_lor_χ_μn!(χ_μm, K, hyper_edge)

        for m ∈ hyper_edge
            χ_μ *= K[m]
        end

        for i ∈ 1:N
            inner_summ = 0.0
            for (hyperedge_idx, m) ∈ enumerate(hyper_edge)
                if c[m, i] ≠ 0
                    inner_summ += c[m, i] * κ[m, i] * K[m] * χ_μm[hyperedge_idx]^2
                end
            end
            du[i] += Γ[μ] * inner_summ
        end

    end
    return du
end

function generate_hyper_graph(M::Int, nbr_edges::Int, average_order::Int, max_order::Int) :: Vector{Vector{Int}}
    numbers = collect(1:M)  # Ensure every number appears at least once
    shuffle!(numbers)  # Randomize order

    # Initialize an empty list of edges
    hyper_graph = [Set{Int}() for _ in 1:nbr_edges]

    # Distribute numbers across edges while ensuring coverage
    for num in numbers
        edge_index = rand(1:nbr_edges)  # Random edge assignment
        push!(hyper_graph[edge_index], num)  # Ensure each number appears at least once
    end

    # Fill remaining edges randomly to approximate average_order
    for i in 1:nbr_edges
        order_size = clamp(round(Int, rand(Normal(average_order, 1))), 1, max_order)
        extra_elements = setdiff(rand(1:M, order_size), hyper_graph[i])
        union!(hyper_graph[i], extra_elements)  # Ensure uniqueness
    end

    # Convert edges back to vectors for final output
    return [collect(edge) for edge in hyper_graph]
end

end # module jlSAT
