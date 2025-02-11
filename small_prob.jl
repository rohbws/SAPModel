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

@info "Defining the optimization model..."
model = Model(Gurobi.Optimizer)

@info "Adding decision variables..."
@variable(model, M[1:F, 1:G], Bin)

@info "Setting the objective function..."
# Objective: Minimize total walking distance for DEPARTING passengers
@objective(model, Min, 
    sum(W_g[g] * P_f[departing_indices[f]] * M[departing_indices[f], g] for f in 1:F_dep, g in 1:G) + sum(W_b[g] * P_f[arriving_indices[f]] * M[arriving_indices[f], g] for f in 1:F_arr, g in 1:G)
)

@info "Adding constraints: Each flight is assigned exactly one gate..."
# Each flight is assigned exactly one gate
@constraint(model, [f in 1:F], sum(M[f, g] for g in 1:G) == 1)

# -------------------------------
# Precompute Conflict Pairs
# -------------------------------

@info "Precomputing conflict pairs..."
# These are pairs of flights (f1,f2) that overlap in time 
# (with a buffer added to the exit time) and belong to different aircraft.
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
# Add constraints: no two conflicting flights may be assigned to the same gate.
for (f1, f2) in conflict_pairs
    for g in 1:G
        @constraint(model, M[f1, g] + M[f2, g] <= 1)
    end
end

# -------------------------------
# Precompute Same‐Gate Pairs for Connections
# -------------------------------

@info "Precomputing same-gate pairs for connections..."
# These are pairs where an arriving flight and a departing flight
# (with the same tail number) must be assigned the same gate 
# if the departing flight’s start time is within 2 hours of the arriving flight’s exit.
same_gate_pairs = Vector{Tuple{Int, Int}}()
for f1 in arriving_indices
    for f2 in departing_indices
        if df.TailNumber[f1] == df.TailNumber[f2] && (df.ExitGateTime[f1] + 120 >= df.EnterGateTime[f2])
            push!(same_gate_pairs, (f1, f2))
        end
    end
end

@info "Adding same-gate constraints..."
# Add same‐gate constraints
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
@info "Optimization complete."
