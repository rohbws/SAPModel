using JuMP, Gurobi, CSV, DataFrames

# Helper function to check if two flights (using their global indices) conflict in gate occupancy.
# If they are on the same tail number, we assume they are connected and thus not in conflict.
function conflicts(global_df, f1, f2, buffer_time)
    # Do not treat flights with the same tail as conflicting.
    if f1 == 50 && f2 == 54
        println("Tail numbers: ", global_df.TailNumber[f1], " ", global_df.TailNumber[f2])
    end
    if global_df.TailNumber[f1] == global_df.TailNumber[f2]
        return false
    end
    enter1 = global_df.ArrivalTimeMinutes[f1]
    exit1  = global_df.OffTimeMinutes[f1] + buffer_time
    enter2 = global_df.ArrivalTimeMinutes[f2]
    exit2  = global_df.OffTimeMinutes[f2] + buffer_time
    return (enter1 < exit2) && (enter2 < exit1)
end

function assign_gates(start_flight, end_flight, flights_per_save = 5, lookahead = 10, departing = true, arriving = true, connecting = true, buffer_time = 0)
    # Load Data
    df = CSV.read("Data/Final_Formatted_Sample_Day.csv", DataFrame)
    walking_distances = CSV.read("Data/Walking Distances Arriving and Departing Pax.csv", DataFrame)
    walking_distances_gate_to_gate = CSV.read("Data/Walking Distances Gate-to-Gate.csv", DataFrame)
    connections_matrix = CSV.read("Data/connections_matrix.csv", DataFrame)
    connections_tier_matrix = CSV.read("Data/connections_tier_matrix.csv", DataFrame) # (i,j) = tier of connection time between flight i and flight j
    # ^ tier 1: 0-45 minute layover | tier 2: 45-90 minute layover | tier 3: >90 minute layover

    # Sort df by time the aircraft enters a gate
    df = df[sortperm(df.ArrivalTimeMinutes), :]

    # Function that solves a single window (from global flight start_flight to end_flight)
    function assign_gates_single(start_flight, end_flight, departing = true, arriving = true, connecting = true, buffer_time = buffer_time, assignments_overall = Dict())
        """
        Solves the gate assignment for a window of flights.
        - start_flight and end_flight are global indices.
        - assignments_overall is a dictionary of locked assignments from previous windows.
        """
        num_flights = end_flight - start_flight + 1
        df_small = df[start_flight:end_flight, :]
        connections_matrix_small = connections_matrix[start_flight:end_flight, start_flight:end_flight]
        connections_tier_matrix_small = connections_tier_matrix[start_flight:end_flight, start_flight:end_flight]
        connections_tier_matrix_small[1:size(connections_tier_matrix_small,1), 1:size(connections_tier_matrix_small,2)-1] = Matrix(connections_tier_matrix_small[1:size(connections_tier_matrix_small,1), 2:end])

        # Build a connections time matrix (adjusted to window indices)
        T_f1_f2 = zeros(num_flights, num_flights)
        T_f1_f2[1:size(connections_matrix_small,1), 1:size(connections_matrix_small,2)-1] =
            Matrix(connections_matrix_small[1:size(connections_matrix_small,1), 2:end])
    
        # Define enter and exit gate times for the window
        df_small[!, :EnterGateTime] = df_small.ArrivalTimeMinutes
        df_small[!, :ExitGateTime]  = df_small.OffTimeMinutes

        # Separate arriving and departing flights (using the window's data)
        departing_indices = findall(df_small.IsDeparting .== "Y")
        arriving_indices  = findall(df_small.IsDeparting .== "N")
        F_dep = length(departing_indices)   # Number of departing flights in the window
        F_arr = length(arriving_indices)    # Number of arriving flights in the window
        F     = num_flights                 # Total flights in window
        G     = 96                          # Number of gates

        # Passenger counts (for objective function)
        P_df = [df_small.PassengersDept[departing_indices[f]] for f in 1:F_dep]
        P_af = [df_small.PassengersArr[arriving_indices[f]]  for f in 1:F_arr]
    
        # Walking distances parameters
        W_g = walking_distances.TSA_to_Gate     # From security to gate
        W_b = walking_distances.Gate_to_Bag       # From gate to baggage claim
        W_g1_g2 = Matrix(walking_distances_gate_to_gate)
    
        # -------------------------------
        # Define the Optimization Model
        # -------------------------------
        model = Model(Gurobi.Optimizer)
    
        # Decision variables: M[f, g] is 1 if flight f (local index) is assigned to gate g.
        @variable(model, M[1:F, 1:G], Bin)
    
        # Process connecting flight pairs (if enabled)
        connecting_pairs = Vector{Tuple{Int, Int}}()
        if connecting   
            # Pre-process connecting flight pairs
            for f1 in 1:F
                for f2 in 1:F
                    if connections_tier_matrix_small[f1, f2] == 1
                        push!(connecting_pairs, (f1, f2))
                    end
                end
            end
            # Auxiliary variables for linearizing M[f1,g1] * M[f2,g2]
            @variable(model, Z[connecting_pairs, 1:G, 1:G], Bin)
        end
    
        @objective(model, Min,
            departing * sum(W_g[g] * P_df[f] * M[departing_indices[f], g] for f in 1:F_dep, g in 1:G)
            + arriving * sum(W_b[g] * P_af[f] * M[arriving_indices[f], g] for f in 1:F_arr, g in 1:G)
            + connecting * sum(
                T_f1_f2[f1, f2] * W_g1_g2[g1, g2] * Z[(f1,f2), g1, g2]
                for (f1,f2) in connecting_pairs, g1 in 1:G, g2 in 1:G
            )
        )
    
        if connecting
            # Linearization constraints for product variables Z = M[f1, g1] * M[f2, g2]
            @constraints(model, begin
                [(f1, f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
                    Z[(f1, f2), g1, g2] <= M[f1, g1]
                [(f1, f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
                    Z[(f1, f2), g1, g2] <= M[f2, g2]
                [(f1, f2) in connecting_pairs, g1 in 1:G, g2 in 1:G], 
                    Z[(f1, f2), g1, g2] >= M[f1, g1] + M[f2, g2] - 1
            end)
        end
    
        # Each flight in the window must be assigned exactly one gate.
        @constraint(model, [f in 1:F], sum(M[f, g] for g in 1:G) == 1)

        # -------------------------------------------------------
        # Incorporate Locked Flight Information from Previous Windows
        # -------------------------------------------------------
        # For each flight that has been locked (with a gate) from previous windows,
        # we adjust the current window as follows.
        # (a) If a flight in the current window (global index = start_flight + local_f - 1)
        #     conflicts (in time) with the locked flight, then it cannot be assigned to the locked gate.
        # (b) If the flight in the current window should use the same gate as the locked flight
        #     (i.e. same tail and close timing), then force that assignment.
        for (global_locked, locked_gate) in assignments_overall
            for local_f in 1:F
                #global_local = start_flight + local_f - 1
                if local_f == 4 && locked_gate == 35
                    println("Locked flight: ", global_locked, " at gate ", locked_gate)
                end
                if conflicts(df, global_locked, local_f, buffer_time)
                    @constraint(model, M[local_f, locked_gate] == 0)
                end
            end
        end

        for (global_locked, locked_gate) in assignments_overall
            for local_f in 1:F
                # Here we use df_small for the current window flight,
                # while using df for the locked flight.
                if ((df[global_locked, :].TailNumber == df_small.TailNumber[local_f]) &&
                    (df[global_locked, :].ArrivalTimeMinutes == df_small.EnterGateTime[local_f]) &&
                    (df[global_locked, :].OffTimeMinutes == df_small.ExitGateTime[local_f]))
                    @constraint(model, M[local_f, locked_gate] == 1)
                end
            end
        end

        # -------------------------------------------------------
        # Constraints Among Flights Within the Current Window
        # -------------------------------------------------------
        # Precompute conflict pairs (for flights that are on different aircraft)
        conflict_pairs = Vector{Tuple{Int, Int}}()
        for f1 in 1:(F-1)
            for f2 in (f1+1):F
                if df_small.TailNumber[f1] != df_small.TailNumber[f2]
                    enter1  = df_small.EnterGateTime[f1]
                    depart1 = df_small.ExitGateTime[f1] + buffer_time
                    enter2  = df_small.EnterGateTime[f2]
                    depart2 = df_small.ExitGateTime[f2] + buffer_time

                    if (f1 == 9 && f2 == 50) || (f1 == 50 && f2 == 9)
                        println("CONFLICT PAIR 1: ", f1, " and ", f2)
                        println("Times: ", enter1, " ", depart1, " ", enter2, " ", depart2)
                    end
                    
                    if (enter1 < depart2) && (enter2 < depart1)
                        push!(conflict_pairs, (f1, f2))
                    end
                end
            end
        end
    
        # For every conflicting pair, ensure they are not assigned the same gate.
        for (f1, f2) in conflict_pairs
            if (f1 == 9 && f2 == 50) || (f1 == 50 && f2 == 9)
                println("CONFLICT PAIR: ", f1, " and ", f2)
            end
            for g in 1:G
                @constraint(model, M[f1, g] + M[f2, g] <= 1)
            end
        end
    
        # Precompute same‐gate pairs among local flights (arriving vs. departing)
        same_gate_pairs = Vector{Tuple{Int, Int}}()
        for f1 in arriving_indices
            for f2 in departing_indices
                if (df_small.TailNumber[f1] == df_small.TailNumber[f2]) &&
                   (df_small.ExitGateTime[f1] == df_small.ExitGateTime[f2]) &&
                   (df_small.EnterGateTime[f1] == df_small.EnterGateTime[f2])
                    push!(same_gate_pairs, (f1, f2))
                end
            end
        end
    
        # Enforce that flights in a same‐gate pair share the same gate.
        for (f1, f2) in same_gate_pairs
            for g in 1:G
                @constraint(model, M[f1, g] == M[f2, g])
            end
        end
    
        # -------------------------------
        # Solve the Model
        # -------------------------------
        optimize!(model)

        println("Termination status: ", termination_status(model))

        # Check infeasibility
        if termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED
            println("Model is infeasible; computing IIS...")

            compute_conflict!(model)

            # Print IIS constraints
            println("The following constraints are in the IIS:")
            for (i, c) in enumerate(all_constraints(model; include_variable_in_set_constraints=false))
                iis_status = MOI.get(model, MOI.ConstraintConflictStatus(), c)
                
                if iis_status == MOI.IN_CONFLICT
                    println("Constraint: ", iis_status)
                    println("Constraint: ", c)
                end
            end
        end
    
        # Extract assignments from the model (using local flight indices)
        assignments = Dict(f => g for f in 1:F, g in 1:G if value(M[f, g]) ≈ 1)
    
        return assignments
    end    

    # Compute the window size and prepare to collect overall assignments.
    flights_in_window = flights_per_save + lookahead
    assignments_overall = Dict()
    num_flights_overall = end_flight - start_flight + 1
    println("Flights locked per window: ", flights_per_save)

    # Slide the window over the global flight indices.
    for i in start_flight:flights_per_save:end_flight
        local_end = min(i + flights_in_window - 1, end_flight)
        local_assignments = assign_gates_single(i, local_end, departing, arriving, connecting, buffer_time, assignments_overall)
        println("Assigned gates for flights ", i, " to ", local_end)
        
        # Convert the local (window) assignment keys to global flight indices.
        global_assignments = Dict((k + i - 1) => v for (k, v) in local_assignments)
        println("Global assignments: ", global_assignments)
        
        # Lock in only the first flights_per_save flights from the current window.
        locked_in_assignments = Dict((k + i - 1) => v for (k, v) in local_assignments if k <= flights_per_save)
        println("Locked assignments for flights ", i, " to ", min(i + flights_per_save - 1, end_flight))
        println("Locked in assignments: ", locked_in_assignments)
        
        # Merge these locked assignments into the overall assignment dictionary.
        assignments_overall = merge(assignments_overall, locked_in_assignments)
    end

    # -------------------------------
    # Write Out Final Assignments
    # -------------------------------
    df[!, :OptDepGate] = Vector{Union{String, Missing}}(missing, nrow(df))
    df[!, :OptArrGate] = Vector{Union{String, Missing}}(missing, nrow(df))

    # Gate mapping (gate number to gate code)
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
    println("Total flights: ", num_flights_overall)
    println("Total locked assignments: ", length(assignments_overall))
    println(assignments_overall)

    # Apply the locked assignments to the global dataframe.
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

    # Save only the window (or the full day, as desired)
    df_window = df[start_flight:end_flight, :]
    CSV.write("Optimized_Gate_Assignments_Sample_Day.csv", df_window)
end

assign_gates(1, 1200, 50, 50, true, false, false)