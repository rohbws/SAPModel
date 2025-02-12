using JuMP, Gurobi, CSV, DataFrames, Logging

# Set up logging
logger = ConsoleLogger(stderr, Logging.Info)
global_logger(logger)

# -------------------------------
# Load Data and Setup Parameters
# -------------------------------

@info "Loading data from CSV files..."
file_path = "Data/May_2nd_Formatted.csv"
df = CSV.read(file_path, DataFrame)

walking_distances_file = "Data/Walking Distances Arriving and Departing Pax.csv"
walking_distances = CSV.read(walking_distances_file, DataFrame)

@info "Separating arriving and departing flights..."
# Separate arriving and departing flights
departing_indices = findall(df.IsDeparting .== "Y")
arriving_indices  = findall(df.IsDeparting .== "N")
F_dep = length(departing_indices)  # Number of departing flights
F_arr = length(arriving_indices)     # Number of arriving flights
F     = nrow(df)                     # Total flights
G     = 96                         # Number of gates

@info "Setting up parameters..."
# Define enter and exit gate times
df[!, :EnterGateTime] = df.ArrivalTimeMinutes
df[!, :ExitGateTime]  = df.OffTimeMinutes

BUFFER_TIME = 0   # Buffer time (modifiable parameter)

# Walking times to gates from TSA
W_g = walking_distances.TSA_to_Gate

# Walking times to gates from baggage claim
W_b = walking_distances.Gate_to_Bag

# Passenger count for flight f (if PassengersArr > 0 then use it, else use PassengersDept)
P_f = [ df.PassengersArr[f] > 0 ? df.PassengersArr[f] : df.PassengersDept[f] for f in 1:F ]

# -------------------------------
# Define the Model
# -------------------------------

batch_size = 100
num_batches = ceil(Int, F / batch_size)
M_total = zeros(Bool, F, G)  # A binary matrix of size F × G

for batch in 1:num_batches
    batch_start = (batch - 1) * batch_size + 1
    batch_end = min(batch * batch_size, F)

    @info "Processing batch $batch ($batch_start to $batch_end)..."
    
    # Define optimization model for the batch
    model = Model(Gurobi.Optimizer)

    @variable(model, M[batch_start:batch_end, 1:G], Bin)

    @objective(model, Min, 
        sum(W_g[g] * P_f[f] * M[f, g] for f in intersect(departing_indices, batch_start:batch_end), g in 1:G) +
        sum(W_b[g] * P_f[f] * M[f, g] for f in intersect(arriving_indices, batch_start:batch_end), g in 1:G)
    )

    @constraint(model, [f in batch_start:batch_end], sum(M[f, g] for g in 1:G) == 1)

    # Compute conflicts only for this batch
    conflict_pairs = Vector{Tuple{Int, Int}}()

    for f1 in batch_start:(batch_end-1)
        for f2 in (f1+1):batch_end
            if df.TailNumber[f1] != df.TailNumber[f2] &&
            df.EnterGateTime[f1] < df.ExitGateTime[f2] + BUFFER_TIME &&
            df.EnterGateTime[f2] < df.ExitGateTime[f1] + BUFFER_TIME
                push!(conflict_pairs, (f1, f2))
            end
        end
    end

    for (f1, f2) in conflict_pairs
        for g in 1:G
            @constraint(model, M[f1, g] + M[f2, g] <= 1)
        end
    end

    # Compute same-gate pairs only for this batch
    same_gate_pairs = Vector{Tuple{Int, Int}}()

    for f1 in arriving_indices
        if batch_start ≤ f1 ≤ batch_end
            for f2 in departing_indices
                if batch_start ≤ f2 ≤ batch_end &&
                df.TailNumber[f1] == df.TailNumber[f2] &&
                df.ExitGateTime[f1] + 120 ≥ df.EnterGateTime[f2]
                    push!(same_gate_pairs, (f1, f2))
                end
            end
        end
    end

    for (f1, f2) in same_gate_pairs
        for g in 1:G
            @constraint(model, M[f1, g] == M[f2, g])
        end
    end

    optimize!(model)

    # Store results directly in M_total
    M_total[batch_start:batch_end, :] .= round.(Int, Matrix(value.(M[batch_start:batch_end, :])))

end

@info "Batch Optimization complete."
@info "Optimizing the entire problem..."

# -------------------------------
# Define the Model
# -------------------------------
model = Model(Gurobi.Optimizer)

# Define the binary decision variable M, using the current values in M_total as a warm start
@variable(model, M[1:F, 1:G], Bin, start_value = M_total)

"""
# Set warm start values for the decision variables using M_total
for f in 1:F
    for g in 1:G
        if M_total[f, g] == 1
            set_start_value(M[f, g], 1)  # If the current value is 1, set the start value to 1
        else
            set_start_value(M[f, g], 0)  # If the current value is 0, set the start value to 0
        end
    end
end
"""

# Objective: Minimize total walking distance for DEPARTING passengers
@objective(model, Min, 
    sum(W_g[g] * P_f[departing_indices[f]] * M[departing_indices[f], g] for f in 1:F_dep, g in 1:G) + 
    sum(W_b[g] * P_f[arriving_indices[f]] * M[arriving_indices[f], g] for f in 1:F_arr, g in 1:G)
)

# Each flight is assigned exactly one gate
@constraint(model, [f in 1:F], sum(M[f, g] for g in 1:G) == 1)

# -------------------------------
# Precompute Conflict Pairs
# -------------------------------
conflict_pairs = Vector{Tuple{Int, Int}}()
for f1 in 1:(F-1)
    for f2 in (f1+1):F
        if df.TailNumber[f1] != df.TailNumber[f2]
            enter1  = df.EnterGateTime[f1]
            depart1 = df.ExitGateTime[f1] + BUFFER_TIME
            enter2  = df.EnterGateTime[f2]
            depart2 = df.ExitGateTime[f2] + BUFFER_TIME
            if (enter1 < depart2) && (enter2 < depart1)
                push!(conflict_pairs, (f1, f2))
            end
        end
    end
end

# Add constraints: no two conflicting flights may be assigned to the same gate.
for (f1, f2) in conflict_pairs
    for g in 1:G
        @constraint(model, M[f1, g] + M[f2, g] <= 1)
    end
end

# -------------------------------
# Precompute Same‐Gate Pairs for Connections
# -------------------------------
same_gate_pairs = Vector{Tuple{Int, Int}}()
for f1 in arriving_indices
    for f2 in departing_indices
        if df.TailNumber[f1] == df.TailNumber[f2] && (df.ExitGateTime[f1] + 120 >= df.EnterGateTime[f2])
            push!(same_gate_pairs, (f1, f2))
        end
    end
end

# Add same‐gate constraints
for (f1, f2) in same_gate_pairs
    for g in 1:G
        @constraint(model, M[f1, g] == M[f2, g])
    end
end

# -------------------------------
# Solve the Model
# -------------------------------
optimize!(model)


# Extract results
assignments = Dict(f => g for f in 1:F, g in 1:G if M_total[f, g] ≈ 1)

# Create new columns for optimized gate assignments
df[!, :OptDepGate] = Vector{Union{String, Missing}}(missing, nrow(df))
df[!, :OptArrGate] = Vector{Union{String, Missing}}(missing, nrow(df))

# Gate mapping dictionary
gate_mapping = Dict(
    1 => "A8", 2 => "A9", 3 => "A10", 4 => "A11", 5 => "A13", 
    6 => "A14", 7 => "A15", 8 => "A16", 9 => "A17", 10 => "A18", 
    11 => "A19", 12 => "A20", 13 => "A21", 14 => "A22", 15 => "A23", 
    16 => "A24", 17 => "A25", 18 => "A28", 19 => "A29", 20 => "A33", 
    21 => "A34", 22 => "A35", 23 => "A36", 24 => "A37", 25 => "A38", 
    26 => "A39", 27 => "B1", 28 => "B2", 29 => "B3", 30 => "B4", 
    31 => "B5", 32 => "B6", 33 => "B7", 34 => "B9", 35 => "B10", 
    36 => "B11", 37 => "B12", 38 => "B14", 39 => "B16", 40 => "B17", 
    41 => "B18", 42 => "B19", 43 => "B21", 44 => "B22", 45 => "B24", 
    46 => "B25", 47 => "B26", 48 => "B27", 49 => "B28", 50 => "B29", 
    51 => "B30", 52 => "B31", 53 => "B32", 54 => "B33", 55 => "B34", 
    56 => "B35", 57 => "B36", 58 => "B37", 59 => "B38", 60 => "B39", 
    61 => "B40", 62 => "B42", 63 => "B43", 64 => "B44", 65 => "B46", 
    66 => "B47", 67 => "B48", 68 => "B49", 69 => "C2", 70 => "C4", 
    71 => "C6", 72 => "C7", 73 => "C8", 74 => "C10", 75 => "C11", 
    76 => "C12", 77 => "C14", 78 => "C15", 79 => "C16", 80 => "C17", 
    81 => "C19", 82 => "C20", 83 => "C21", 84 => "C22", 85 => "C24", 
    86 => "C26", 87 => "C27", 88 => "C28", 89 => "C29", 90 => "C30", 
    91 => "C31", 92 => "C33", 93 => "C35", 94 => "C36", 95 => "C37", 
    96 => "C39"
)

# Assign gates
for f in 1:F
    gate_number = get(assignments, f, missing)
    if !ismissing(gate_number)
        gate_code = get(gate_mapping, gate_number, missing)
        if df.IsDeparting[f] == "Y"
            df[f, :OptDepGate] = gate_code
        else
            df[f, :OptArrGate] = gate_code
        end
    end
end

println(df)

# Save results
CSV.write("Optimized_Gate_Assignments_Sample_Day.csv", df)
