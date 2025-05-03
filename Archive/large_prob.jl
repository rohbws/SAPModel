using JuMP, Gurobi, CSV, DataFrames, Logging

# -------------------------------
# Load Data and Setup Parameters
# -------------------------------

@info "Loading data and setting up parameters..."
file_path = "Data/Small_May_2nd_Formatted.csv"
df = CSV.read(file_path, DataFrame)

walking_distances_file1 = "Data/Walking Distances Arriving and Departing Pax.csv"
walking_distance_file2 = "Data/Walking Distances Gate-to-Gate.csv"
walking_distances = CSV.read(walking_distances_file1, DataFrame)
walking_distances_gate_to_gate = CSV.read(walking_distance_file2, DataFrame)

connections_matrix_file = "Data/small_connections_matrix.csv"
connections_matrix = CSV.read(connections_matrix_file, DataFrame)

@info "Separating arriving and departing flights..."
departing_indices = findall(df.IsDeparting .== "Y")
arriving_indices  = findall(df.IsDeparting .== "N")
F_dep = length(departing_indices)   # Number of departing flights
F_arr = length(arriving_indices)    # Number of arriving flights
F     = nrow(df)                    # Total flights
G     = 96                          # Number of gates

@info "Defining enter and exit gate times..."
df[!, :EnterGateTime] = df.ArrivalTimeMinutes
df[!, :ExitGateTime]  = df.OffTimeMinutes

BUFFER_TIME = 0   # Buffer time (modifiable parameter)

@info "Calculating passenger counts..."
P_df = [df.PassengersDept[departing_indices[f]] for f in 1:F_dep]  # Departing pax
P_af = [df.PassengersArr[arriving_indices[f]]  for f in 1:F_arr]   # Arriving pax

@info "Initializing transfer passengers matrix..."
T_f1_f2 = zeros(F, F)  # Initialize F×F matrix
T_f1_f2[1:size(connections_matrix,1), 1:size(connections_matrix,2)-1] = Matrix(connections_matrix[1:size(connections_matrix,1), 2:end])

@info "Loading walking distances..."
W_g       = walking_distances.TSA_to_Gate     # Gate distance from security
W_b       = walking_distances.Gate_to_Bag     # Gate distance to baggage claim
W_g1_g2   = zeros(G, G)  # Initialize G×G matrix
W_g1_g2[1:95, 1:95] = Matrix(walking_distances_gate_to_gate[1:95, 2:end])  # Fill first 95×95 elements

@info "Defining weights..."
lambda_ = 1.0
alpha   = 1.0

@info "Pre-processing connecting flight pairs..."
connecting_pairs = Vector{Tuple{Int, Int}}()
for f1 in 1:F
    for f2 in 1:F
        if T_f1_f2[f1, f2] > 0
            push!(connecting_pairs, (f1, f2))
        end
    end
end

# -------------------------------
# Define the Model
# -------------------------------
@info "Defining the optimization model..."
model = Model(Gurobi.Optimizer)

# Set up logging to both terminal and file
set_optimizer_attribute(model, "LogFile", "threeobj_log.txt")


@info "Defining gate-assignment decision variables..."
@variable(model, M[1:F, 1:G], Bin)

@info "Defining auxiliary variables for linearizing M[f1,g1] * M[f2,g2]..."
@variable(model, Z[connecting_pairs, 1:G, 1:G], Bin)

@info "Defining the objective function..."
@objective(model, Min,
    sum(W_g[g] * P_df[f] * M[departing_indices[f], g] for f in 1:F_dep, g in 1:G)
    + lambda_ * sum(W_b[g] * P_af[f] * M[arriving_indices[f], g] for f in 1:F_arr, g in 1:G)
    + alpha * sum(
        T_f1_f2[f1, f2] * W_g1_g2[g1, g2] * Z[(f1,f2), g1, g2]
        for (f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G
    )
)

@info "Adding linearization constraints for Z = M[f1,g1] * M[f2,g2]..."
@constraints(model, begin
    [(f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
        Z[(f1,f2),g1,g2] <= M[f1,g1]
    [(f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
        Z[(f1,f2),g1,g2] <= M[f2,g2]
    [(f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
        Z[(f1,f2),g1,g2] >= M[f1,g1] + M[f2,g2] - 1
end)

@info "Adding constraints to ensure each flight is assigned to exactly one gate..."
@constraint(model, [f in 1:F], sum(M[f, g] for g in 1:G) == 1)

# -------------------------------
# Precompute Conflict Pairs
# -------------------------------
@info "Precomputing conflict pairs..."
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

@info "Adding constraints to ensure no two conflicting flights share the same gate..."
for (f1, f2) in conflict_pairs
    for g in 1:G
        @constraint(model, M[f1, g] + M[f2, g] <= 1)
    end
end

# -------------------------------
# Precompute Same‐Gate Pairs
# -------------------------------
@info "Precomputing same-gate pairs..."
same_gate_pairs = Vector{Tuple{Int, Int}}()
for f1 in arriving_indices
    for f2 in departing_indices
        if df.TailNumber[f1] == df.TailNumber[f2] &&
           (df.ExitGateTime[f1] + 120 >= df.EnterGateTime[f2])
            push!(same_gate_pairs, (f1, f2))
        end
    end
end

@info "Adding constraints to ensure same tail number and close arrival/departure flights use the same gate..."
for (f1, f2) in same_gate_pairs
    for g in 1:G
        @constraint(model, M[f1, g] == M[f2, g])
    end
end


# -------------------------------
# Solve the Model
# -------------------------------
@info "Solving the optimization model..."
optimize!(model)
@info "Optimization complete."


# Extract results
assignments = Dict(f => g for f in 1:F, g in 1:G if value(M[f, g]) ≈ 1)

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
CSV.write("ThreeObj_Optimized_Gate_Assignments_Sample_Day.csv", df)