using JuMP, Gurobi, CSV, DataFrames, Logging

# -------------------------------
# Load Data and Setup Parameters
# -------------------------------

@info "Loading data from CSV files..."
file_path = "Data/Small_May_2nd_Formatted.csv"
df = CSV.read(file_path, DataFrame)

walking_distances_file = "Data/Walking Distances Arriving and Departing Pax.csv"
walking_distances = CSV.read(walking_distances_file, DataFrame)

@info "Separating arriving and departing flights..."
departing_indices = findall(df.IsDeparting .== "Y")
arriving_indices  = findall(df.IsDeparting .== "N")
F_dep = length(departing_indices)  # Number of departing flights
F_arr = length(arriving_indices)     # Number of arriving flights
F     = nrow(df)                     # Total flights
G     = 96                         # Number of gates

@info "Defining enter and exit gate times..."
df[!, :EnterGateTime] = df.ArrivalTimeMinutes
df[!, :ExitGateTime]  = df.OffTimeMinutes

BUFFER_TIME = 0   # Buffer time (modifiable parameter)

@info "Loading walking times..."
W_g = walking_distances.TSA_to_Gate
W_b = walking_distances.Gate_to_Bag

@info "Calculating passenger counts..."
P_f = [ df.PassengersArr[f] > 0 ? df.PassengersArr[f] : df.PassengersDept[f] for f in 1:F ]

# -------------------------------
# Define the Model
# -------------------------------

@info "Defining the optimization model..."
model = Model(Gurobi.Optimizer)
@variable(model, M[1:F, 1:G], Bin)

@info "Setting the objective function..."
@objective(model, Min, 
    sum(W_g[g] * P_f[departing_indices[f]] * M[departing_indices[f], g] for f in 1:F_dep, g in 1:G)
)

@info "Adding constraints: each flight is assigned exactly one gate..."
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

@info "Adding conflict constraints..."
for (f1, f2) in conflict_pairs
    for g in 1:G
        @constraint(model, M[f1, g] + M[f2, g] <= 1)
    end
end

# -------------------------------
# Precompute Same‐Gate Pairs for Connections
# -------------------------------

@info "Precomputing same-gate pairs for connections..."
same_gate_pairs = Vector{Tuple{Int, Int}}()
for f1 in arriving_indices
    for f2 in departing_indices
        if df.TailNumber[f1] == df.TailNumber[f2] && (df.ExitGateTime[f1] + 120 >= df.EnterGateTime[f2])
            push!(same_gate_pairs, (f1, f2))
        end
    end
end

@info "Adding same-gate constraints..."
for (f1, f2) in same_gate_pairs
    for g in 1:G
        @constraint(model, M[f1, g] == M[f2, g])
    end
end

# -------------------------------
# Solve the Model
# -------------------------------

@info "Optimizing the model..."
optimize!(model)

@info "Retrieving the optimal objective value..."
opt_obj = JuMP.objective_value(model)

@info "Computing average walking distance for departing passengers..."
total_departing_passengers = sum(df.PassengersDept)
avg_departing_wd = opt_obj / total_departing_passengers

@info "Determining the assigned gate for each flight..."
F, G = size(M)
assigned_gate = Vector{Int}(undef, F)
for f in 1:F
    assigned_gate[f] = findfirst(g -> JuMP.value(M[f, g]) >= 0.5, 1:G)
end

@info "Computing average walking distance for departing passengers..."
departing_indices = findall(x -> x == "Y", df.IsDeparting)
total_departing_passengers = sum(df.PassengersDept[departing_indices])
total_departing_wd = 0.0
for f in departing_indices
    gate = assigned_gate[f]
    idx = findfirst(==(gate), walking_distances.Gate_Int)
    TSA_distance = walking_distances.TSA_to_Gate[idx]
    total_departing_wd += df.PassengersDept[f] * TSA_distance
end
avg_departing_wd = total_departing_wd / total_departing_passengers

@info "Computing average walking distance for arriving passengers..."
arrival_indices = findall(x -> x == "N", df.IsDeparting)
total_arriving_passengers = sum(df.PassengersArr[arrival_indices])
total_arriving_wd = 0.0
for f in arrival_indices
    gate = assigned_gate[f]
    idx = findfirst(==(gate), walking_distances.Gate_Int)
    baggage_distance = walking_distances.Gate_to_Bag[idx]
    total_arriving_wd += df.PassengersArr[f] * baggage_distance
end
avg_arriving_wd = total_arriving_wd / total_arriving_passengers

@info "Computing average walking distance for connecting passengers..."
total_connection_wd = 0.0
total_connection_passengers = 0

walking_distance_file = "Data/Walking Distances Gate-to-Gate.csv"
walking_distances = CSV.read(walking_distance_file, DataFrame)
walking_distances_gate_to_gate = CSV.read(walking_distance_file, DataFrame)

connections_matrix_file = "Data/small_connections_matrix.csv"
conn_mat = CSV.read(connections_matrix_file, DataFrame)
conn_mat = Matrix(conn_mat)
for i in 2:size(conn_mat, 1)
    for j in 2:size(conn_mat, 2)
        num_connect = conn_mat[i, j]
        if num_connect > 0
            gate_i = assigned_gate[i]
            gate_j = assigned_gate[j]
            gate_to_gate_wd = walking_distances_gate_to_gate[gate_i, gate_j]
            total_connection_wd += num_connect * gate_to_gate_wd
            total_connection_passengers += num_connect
        end
    end
end
avg_connection_wd = total_connection_wd / total_connection_passengers

@info "Printing the computed metrics..."
println("Optimal Objective Value: ", opt_obj)
println("Average Departing Passenger Walking Distance: ", avg_departing_wd)
println("Average Arriving Passenger Walking Distance: ", avg_arriving_wd)
println("Average Connecting Passenger Walking Distance: ", avg_connection_wd)
println("Total Departing Passengers: ", total_departing_passengers)
println("Total Arriving Passengers: ", total_arriving_passengers)
println("Total Connection Passengers: ", total_connection_passengers)
