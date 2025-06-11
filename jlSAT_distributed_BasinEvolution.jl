# Copyright (c) 2025 Àron Vízkeleti
# All rights reserved.
#
# This code is protected under copyright law.
# Unauthorized copying, modification, or distribution is prohibited.



using Distributed
using MsgPack
using CodecZlib

@everywhere using jlSAT

function compute_solutions_parallel(initial_conditions, params)
    return pmap(x -> jlSAT.get_basin_evolution(x, params, false), initial_conditions)
end

function populate_initial_conditions!(initial_s, resulting_array, s1_range, s2_range, idx_1::Int, idx_2::Int)
    N = length(initial_s)

    idx = 1
    for coordinate_x in s1_range
        for coordinate_y in s2_range
            # Replace the coordinates at idx_1 and idx_2 with coordinate_x and coordinate_y, respectively
            updated_s = copy(initial_s)  # Create a copy to avoid modifying the original
            updated_s[idx_1] = coordinate_x
            updated_s[idx_2] = coordinate_y
            resulting_array[idx] = updated_s  # Store the updated vector in the resulting array
            idx += 1
        end
    end
end

# How to run effectively
#nohup julia -p 18 -q --project jlSAT_BB_crc.jl > output.log 2>&1 &



# Load required parameters here
problem_name = "uf20-03"
problem_path = "SAT_problems/SATLIB_Benchmark_problems_N20/"*problem_name*".cnf"

img_width = 512
idx_1 = 3
idx_2 = 6
#spin_1_lower = -1.25
#spin_1_upper = 1.25
#spin_2_lower = -1.25
#spin_2_upper = 1.25
spin_1_lower = -5.0
spin_1_upper = 2.5
spin_2_lower = -5.0
spin_2_upper = 3.0

c = jlSAT.load_cnf(problem_path)
M, N = size(c)
HG = [[i] for i ∈ 1:M]
k_factors = 2 .^ (-1.0 * map(row -> count(x -> x != 0, row), eachrow(c)))

parameters = Dict(
    "c" => c,
    "HG" => HG,
    "k_factors" => k_factors,
    "t_max" => 300.0,
    "idx_1" => idx_1,
    "idx_2" => idx_2, 
    "spin_1_lower" => spin_1_lower,
    "spin_1_upper" => spin_1_upper,
    "spin_2_lower" => spin_2_lower,
    "spin_2_upper" => spin_2_upper
)

initial_s = 2 .* rand(N) .- ones(N)
initial_s = [-0.19901167992301128,0.3115297969204447,-1.25,-0.5298622488900302,0.5009323858507226,-1.25,-0.9326560168103919,-0.6835008199249295,-0.2319916576386578,0.4463661269102024,0.6207724720995764,0.9062288787393309,0.4371741936675222,-0.15585194112259715,-0.6809818174459419,0.7155981180326498,0.2180523929628977,-0.2459807327585135,-0.08679053164098494,0.3424679936431947]
#initial_s = [-0.3862867580308902, 0.6179515237649866, -0.7710108728055132, -2.0, -2.5, 0.8957831811433916, -0.6510415142924701, 0.9306113687826127, 0.48355969395539966, -0.29102569232916387, 0.326719800371408, 0.7863778335712621, -0.6046353041781349, 0.083938419006365, -0.22096790275674705, -0.06402579365195304, -0.836171503687982, 0.6271844317392132, 0.09270356410922087, 0.7070397843878766]
s1_range, s2_range = LinRange(spin_1_lower, spin_1_upper, img_width), LinRange(spin_2_lower, spin_2_upper, img_width)
initial_conditions = Vector{Vector{Float64}}(undef, img_width * img_width)
populate_initial_conditions!(initial_s, initial_conditions, s1_range, s2_range, idx_1, idx_2)

# Run parallel computation
results = compute_solutions_parallel(initial_conditions, parameters)

# Write all results to a JSON file (on the main process)
#open("results/"*problem_name*"_basin_evolution_results_s$idx_1 _$spin_1_lower - $spin_1_upper s$idx_2 _$spin_2_lower - $spin_2_upper _dpi$img_width.json", "w") do io
#    JSON3.write(io, results)
#end

# Save results using MessagePack + Gzip compression
open("results/"*problem_name*"_basin_evolution_results_s$idx_1 _$spin_1_lower - $spin_1_upper s$idx_2 _$spin_2_lower - $spin_2_upper _dpi$img_width.mp.gz", "w") do io
    gz = GzipCompressorStream(io)
    write(gz, pack(results))  
    close(gz)
end

println("Results saved successfully!")