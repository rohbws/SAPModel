using JuMP, Gurobi, CSV, DataFrames

BUFFER_TIME = 0

function assign_gates(start_flight, end_flight, flights_per_save = 30, lookahead = 30, departing = true, arriving = true, connecting = true)
    df = CSV.read("Data/Final_Formatted_Sample_Day.csv", DataFrame)
    walking_distances = CSV.read("Data/Walking Distances Arriving and Departing Pax.csv", DataFrame)
    walking_distances_gate_to_gate = CSV.read("Data/Walking Distances Gate-to-Gate.csv", DataFrame)
    connections_matrix = CSV.read("Data/connections_matrix.csv", DataFrame)
    
    function assign_gates_single(start_flight, end_flight, departing = true, arriving = true, connecting = true, buffer_time = 0)
        """
        This function assigns flights to gates based on the input parameters.
        
        Parameters:
        - num_flights: number of flights to pull from dataset.
        - departing: boolean variable. True if you want to minimize departing passengers' walking distance. False otherwise.
        - arriving: boolean variable. True if you want to minimize arriving passengers' walking distance. False otherwise.
        - connecting: boolean variable. True if you want to minimize connecting passengers' walking distance. False otherwise.
        - buffer_time: enforced buffer time between gate occupancies in minutes. Default is 0.
        
        Saves
        - A csv with optimal gate assignments
    
        """
        # -------------------------------
        # Load Data and Setup Parameters
        # -------------------------------
        num_flights = end_flight - start_flight + 1
        
        df_small = df[start_flight:end_flight, :]
        connections_matrix_small = connections_matrix[start_flight:end_flight, start_flight:end_flight]

        T_f1_f2 = zeros(num_flights,num_flights)
        T_f1_f2[1:size(connections_matrix_small,1), 1:size(connections_matrix_small,2)-1] = Matrix(connections_matrix_small[1:size(connections_matrix_small,1), 2:end])
    
        # Separate arriving and departing flights
        departing_indices = findall(df_small.IsDeparting .== "Y")
        arriving_indices  = findall(df_small.IsDeparting .== "N")
        F_dep = length(departing_indices)   # Number of departing flights
        F_arr = length(arriving_indices)    # Number of arriving flights
        F     = num_flights                 # Total flights
        G     = 96                          # Number of gates
    
        # Define enter and exit gate times
        df_small[!, :EnterGateTime] = df_small.ArrivalTimeMinutes
        df_small[!, :ExitGateTime]  = df_small.OffTimeMinutes
    
        # Passenger counts
        P_df = [df_small.PassengersDept[departing_indices[f]] for f in 1:F_dep]  # Departing pax
        P_af = [df_small.PassengersArr[arriving_indices[f]]  for f in 1:F_arr]   # Arriving pax
    
        # Walking distances
        W_g = walking_distances.TSA_to_Gate     # Gate distance from security
        W_b = walking_distances.Gate_to_Bag     # Gate distance to baggage claim
        W_g1_g2 = Matrix(walking_distances_gate_to_gate)
    
        # -------------------------------
        # Define the Model
        # -------------------------------
        model = Model(Gurobi.Optimizer)
    
        # Gate‐assignment decision variables
        @variable(model, M[1:F, 1:G], Bin)
    
        connecting_pairs = Vector{Tuple{Int, Int}}()
        if connecting   
            # Pre-process connecting flight pairs
            for f1 in 1:F
                for f2 in 1:F
                    if T_f1_f2[f1, f2] > 0
                        push!(connecting_pairs, (f1, f2))
                    end
                end
            end
            # Auxiliary variables for linearizing M[f1,g1] * M[f2,g2]
            @variable(model, Z[connecting_pairs, 1:G, 1:G], Bin)
        end
    
        # Minimize everybody's walking distance
        @objective(model, Min,
            departing * sum(W_g[g] * P_df[f] * M[departing_indices[f], g] for f in 1:F_dep, g in 1:G)
            + arriving * sum(W_b[g] * P_af[f] * M[arriving_indices[f], g] for f in 1:F_arr, g in 1:G)
            + connecting * sum(
                T_f1_f2[f1, f2] * W_g1_g2[g1, g2] * Z[(f1,f2), g1, g2]
                for (f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G
            )
        )
    
        if connecting
            # Linearization constraints
            @constraints(model, begin
                [(f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
                    Z[(f1,f2),g1,g2] <= M[f1,g1]
                [(f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
                    Z[(f1,f2),g1,g2] <= M[f2,g2]
                [(f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
                    Z[(f1,f2),g1,g2] >= M[f1,g1] + M[f2,g2] - 1
            end)
        end
    
        # Each flight assigned to exactly one gate
        @constraint(model, [f in 1:F], sum(M[f, g] for g in 1:G) == 1)
        # -------------------------------
        # Precompute Conflict Pairs
        # -------------------------------
        conflict_pairs = Vector{Tuple{Int, Int}}()
        for f1 in 1:(F-1)
            for f2 in (f1+1):F
                if df_small.TailNumber[f1] != df_small.TailNumber[f2]
                    enter1  = df_small.EnterGateTime[f1]
                    depart1 = df_small.ExitGateTime[f1] + BUFFER_TIME
                    enter2  = df_small.EnterGateTime[f2]
                    depart2 = df_small.ExitGateTime[f2] + BUFFER_TIME
                    if (enter1 < depart2) && (enter2 < depart1)
                        push!(conflict_pairs, (f1, f2))
                    end
                end
            end
        end
    
        # No two conflicting flights may share the same gate
        for (f1, f2) in conflict_pairs
            for g in 1:G
                @constraint(model, M[f1, g] + M[f2, g] <= 1)
            end
        end
        # -------------------------------
        # Precompute Same‐Gate Pairs
        # -------------------------------
        same_gate_pairs = Vector{Tuple{Int, Int}}()
        for f1 in arriving_indices
            for f2 in departing_indices
                if df_small.TailNumber[f1] == df_small.TailNumber[f2] &&
                (df_small.ExitGateTime[f1] + 120 >= df_small.EnterGateTime[f2])
                    push!(same_gate_pairs, (f1, f2))
                end
            end
        end
    
        # If same tail number and close arrival/departure, force same gate
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
        assignments = Dict(f => g for f in 1:F, g in 1:G if value(M[f, g]) ≈ 1)
    
        return assignments
    end    

    flights_per_block = flights_per_save + lookahead
    assignments_overall = Dict()
    num_flights_overall = end_flight - start_flight + 1
    println("per save: ", flights_per_save)

    for i in start_flight:flights_per_save:end_flight
        assignments_small = assign_gates_single(i, min(i+flights_per_block, end_flight), departing, arriving, connecting)
        assignments_small = Dict((k + i - 1) => v for (k, v) in assignments_small)
        println("assignments small, ", assignments_small)
        assignments_overall = merge(assignments_overall, assignments_small)
    end
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
    println("flights overall, ", num_flights_overall)
    println("lengs of assignments overall, ", length(assignments_overall))
    println(assignments_overall)
    # Assign gates
    for f in start_flight:end_flight
        gate_number = get(assignments_overall, f, missing)
        if !ismissing(gate_number)
            gate_code = get(gate_mapping, gate_number, missing)
            if df.IsDeparting[f] == "Y"
                df[f, :OptDepGate] = gate_code
            else
                df[f, :OptArrGate] = gate_code
            end
        end
    end

    # Save results
    CSV.write("Optimized_Gate_Assignments_Sample_Day.csv", df)
end

assign_gates(5, 120,10, 30, true,true,true)