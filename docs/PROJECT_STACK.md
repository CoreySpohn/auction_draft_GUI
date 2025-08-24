Project Goal

The project is a next-generation fantasy football auction draft tool designed to provide a significant strategic advantage over existing tools. It moves beyond deterministic projections to a probabilistic, game-theoretic approach that simulates market behavior and optimizes for championship probability rather than just total points.

Full-Stack Architecture

The system is a modern, decoupled web application composed of several specialized layers.

    Frontend: A dynamic web interface built with React (and TypeScript).

Backend API: A Python backend using the FastAPI framework to handle user management, orchestration, and API requests.

High-Performance Core: A C++ engine that runs a multi-threaded, Agent-Based Model (ABM) to simulate the draft market at high speed.

Bridge: nanobind connects the Python backend to the C++ core, allowing for efficient communication with minimal overhead.

Database: PostgreSQL serves as the database to store user data, league information, and the pre-calculated projection and covariance matrices. 

SQLAlchemy will be used in the Python layer as the Object-Relational Mapper (ORM) to interact with the database.

Optimization: Google OR-Tools (CP-SAT) is used in Python to solve the complex problem of finding the optimal roster construction based on the model's outputs.

Hybrid Model for Live Draft Syncing

To accommodate fantasy platforms like ESPN that lack a stable, official API, the project will use a hybrid model combining a web application and a browser extension.

    Web Application (The "Workshop"): This is the primary interface where users perform all their pre-draft preparation. They log in to their accounts, manage league settings, adjust player projections, and review analysis. All this data is saved to their user profile via the backend API.

    Browser Extension (The "Live Assistant"): For live drafts, the user will utilize a Chrome/Firefox extension. This extension will:

        Authenticate the user by communicating with the backend API.

        Scrape the live draft webpage (e.g., espn.com) for real-time events like nominations and bids.

        Communicate with the backend API, sending it scraped data and requesting analysis for the current player.

        Inject a UI directly onto the fantasy site's page to display the tool's advice, such as the calculated maximum bid.

This approach keeps the computationally heavy lifting and data storage centralized on the server while using the extension as a lightweight tool for syncing and displaying information on platforms without API access.

Browser Extension Technology

The browser extension's UI will also be built using React. Its core components are:

    Manifest (manifest.json): The configuration file that defines the extension's permissions and scripts.

    Popup: The UI that appears when clicking the extension icon, used for login.

    Content Scripts: JavaScript that runs directly on the fantasy draft webpage to handle the scraping of draft data and the injection of the tool's UI.