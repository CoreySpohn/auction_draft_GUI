Development Roadmap: Auction Draft Tool Refactoring

This roadmap outlines the six key stages for refactoring the existing auction draft tool. The primary goal is to incrementally replace the monolithic core with a modular, high-performance backend, using the "Strangler Fig" pattern. This will result in a portable and powerful analysis engine ready for future integration with any UI.

Stage 1: Foundation & Hybrid Build System

Objective: Establish a robust, unified build system that seamlessly compiles the C++ core and packages it with the Python application.

Key Activities:

    Restructure Project:

        Create the new directories inside your existing package folder: package/core, package/services, and package/simulation_core.

        Move all C++-related source files into package/simulation_core.

    Configure Unified Build System:

        Create a pyproject.toml file in the project root (auction_draft_GUI/).

        Configure it to use scikit-build-core as the build backend.

        Set the CMake source directory: tool.scikit-build.cmake.source-dir = "package/simulation_core".

        Specify the Python package location: tool.scikit-build.wheel.packages = ["package"].

    Implement CMake Build Script:

        Create the CMakeLists.txt file inside package/simulation_core/ using the template you provided.

        Ensure the final install directive correctly places the compiled module in the right location:
        CMake

        install(
            TARGETS simulation_core_ext
            LIBRARY DESTINATION package/simulation_core
        )

    Create "Hello, World" Bridge:

        In package/simulation_core/src/bindings.cpp, implement a minimal nanobind module with a simple add(a, b) function to verify the toolchain.

Acceptance Criteria:

    The entire project can be installed in an editable mode from the root directory with a single command: pip install -e..

    A test Python script can successfully import package.simulation_core.simulation_core_ext and receive the correct result from the C++ add function.

Stage 2: Core Data API & Probabilistic Bridge

Objective: Define the data contract between Python and C++ and implement the ProjectionAdapter to bridge the gap between your deterministic data and the new probabilistic engine.

Key Activities:

    Define C++ Data Structures:

        In package/simulation_core/include/DataStructures.h, define the core C++ structs: Player, AgentUtilityParams, and AgentState.

    Expose Structs with nanobind:

        In bindings.cpp, use nb::class_ to bind the C++ structs, making them available as Python classes (e.g., CppPlayer).

        Include the necessary nanobind STL headers (<nanobind/stl/vector.h>, <nanobind/stl/map.h>) to handle automatic type conversions.

    Build the ProjectionAdapter:

        Create the ProjectionAdapter class in package/services/projection_adapter.py.

        Implement its primary method to perform a lightweight Monte Carlo simulation. This method will take a deterministic player projection, decompose it into per-game stats, and use historical weekly standard deviations to generate a distribution of season-long outcomes, ultimately producing the required perf_mean and perf_variance.   

Stat Category	

Historical Weekly Std. Dev.  

Passing Yards	82.1 yards
Passing TDs	0.8 TDs
Rushing Yards	11.3 yards
Rushing TDs	0.4 TDs
Receptions	11.1 receptions
Receiving Yards	15.9 yards
Receiving TDs	0.4 TDs

Acceptance Criteria:

    The ProjectionAdapter can successfully take a deterministic projection and output a player object with perf_mean and perf_variance.

    Python code can create and populate lists of CppPlayer and CppAgentState objects, ready to be passed to C++.

Stage 3: Isolate Python Orchestration Logic

Objective: Decouple the core business logic from the UI controller (mainwindow.py) by migrating it into new, dedicated service and core classes.

Key Activities:

    Create DraftStateManager:

        Implement the DraftStateManager class in package/services/draft_state_manager.py.

        Systematically move all state-tracking logic (budgets, rosters, drafted players, etc.) from mainwindow.py into this class.

    Create AnalysisEngine:

        Implement the AnalysisEngine class in package/core/analysis_engine.py.

        Move the logic responsible for triggering analysis and calculating recommendations into this class. Initially, its methods will simply call the old, existing functions.

    Refactor mainwindow.py:

        Modify mainwindow.py to act as a pure controller.

        Instead of managing state and running calculations itself, it will now instantiate DraftStateManager and AnalysisEngine and delegate calls to them (e.g., self.state_manager.update_on_pick(...), self.analysis_engine.get_max_bid(...)).

Acceptance Criteria:

    The mainwindow.py file is significantly smaller and free of business logic.

    The application remains 100% functional, with the UI now driven by the new, isolated backend classes.

Stage 4: Build the C++ Simulation Core

Objective: Implement the high-performance, multi-threaded agent-based simulation engine in C++.

Key Activities:

    Implement the Agent Class:

        Create Agent.h and Agent.cpp in package/simulation_core/.

        Implement the agent's utility function, incorporating risk aversion: Value = perf_mean - (risk_aversion * perf_variance).

        Implement a fast, greedy algorithm for the estimate_best_roster_value heuristic to guide agent bidding.

    Build the Multi-threaded SimulationRunner:

        Create a SimulationRunner class to manage a single auction simulation.

        Use the C++ standard library (<thread>, <mutex>) to create a pool of worker threads that can run thousands of SimulationRunner instances in parallel.

        Ensure result aggregation is thread-safe using a std::mutex.

    Expose the Main Entry Point:

        Implement the run_eoc_analysis C++ function that will be called from Python. This function will orchestrate the multi-threaded "Win" and "Lose" simulations and return the results.

        Expose this function in bindings.cpp.

Acceptance Criteria:

    The compiled C++ module provides a function that can accept a complete draft state from Python, execute a high-speed, multi-threaded simulation, and return the analysis results.

Stage 5: The Switchover

Objective: Activate the new C++ simulation core, replacing the old analysis logic entirely.

Key Activities:

    Integrate C++ Core:

        In AnalysisEngine.run_eoc_analysis, remove the call to the old analysis function.

        Add the new logic:

            Get the current state from DraftStateManager.

            Use ProjectionAdapter to prepare the list of CppPlayer and CppAgentState objects.

            Call package.simulation_core.simulation_core_ext.run_eoc_analysis(...) with the prepared data.

            Process the returned results to calculate the final max bid.

    Validate:

        Run the old and new systems in parallel on test data to ensure the new engine produces reasonable and directionally correct outputs compared to the legacy system.

Acceptance Criteria:

    The tool's real-time bidding advice is now fully powered by the high-performance C++ engine.

    The old, legacy analysis code is no longer called and can be safely removed.

Stage 6: Add Strategic Roster Optimization

Objective: Implement the forward-looking CP-SAT optimizer to provide robust, strategic advice on optimal roster construction.

Key Activities:

    Add Dependency:

        Add ortools-sat to the dependencies list in your pyproject.toml file.

    Implement the Optimizer:

        Create a new method, run_robust_optimization, in the AnalysisEngine class.

        Use Google's OR-Tools library to build the CP-SAT model:

            Variables: Create a NewBoolVar() for each available player.

            Constraints:

                Budget: Sum of player prices must be less than or equal to the remaining budget. Use a high percentile (e.g., 75th) of simulated prices from the C++ core to ensure the plan is robust.

                Roster Slots: Sum of players at each position must equal the number of open slots.

            Objective: Maximize the total Value Over Replacement Player (VORP) of the selected roster.   

    Connect to UI:

        Add a new button or trigger in the UI that calls the analysis_engine.run_robust_optimization() method and displays the recommended list of players to the user.

Acceptance Criteria:

    The user can, at any point during the draft, request a recommendation for the optimal roster to target, providing powerful strategic guidance beyond single-player bidding advice.