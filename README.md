# SDN-MANET Reactive Firewall Simulators

This repository provides a simulation framework for evaluating reactive firewall mechanisms in Software-Defined Networking (SDN) enabled Mobile Ad Hoc Networks (MANETs). The simulator supports realistic mobility scenarios, attack simulations (TCP/UDP flood attacks), and joint optimization for controller placement and IDS (Intrusion Detection System) enablement. This work is associated with the following publication:  
[IEEE Publication – Document 10461418](https://ieeexplore.ieee.org/document/10461418)

## Features

- **Network Topology & Mobility Simulation:**  
  Create custom MANET topologies with various node counts and mobility models (e.g., random, convoy, small teams). NS_movement scenario files (e.g., `Company1_24_nodes_6100-6400.ns_movements`, `Company1_24_nodes_870-1280.ns_movements`) are provided to emulate realistic movements.

- **Reactive Firewall Mechanisms:**  
  Simulate a reactive firewall that detects and mitigates malicious traffic (e.g., TCP/UDP flood attacks) by dynamically applying flow rules and blocking harmful traffic.

- **Attack Simulation:**  
  Emulate TCP/UDP flood attacks to assess the performance and responsiveness of the reactive firewall under various stress conditions.

- **Optimization Algorithms (JCPIE):**  
  Implement joint optimization for selecting the best controller and IDS placements in the network to minimize mitigation delays and maximize network coverage.

- **Performance Evaluation:**  
  Measure key performance metrics such as detection delay, alerting delay, blocking delay, and total mitigation delay to assess the effectiveness of the reactive firewall.

- **Visualization & Analysis:**  
  Generate graphs and animations to visualize network topologies, node connectivity, and simulation results.

## Prerequisites

- **Python 3.x**
- **Required Libraries:**
  - networkx
  - numpy
  - matplotlib
  - (Plus standard libraries: math, time, random, itertools, heapq, collections, etc.)
- **NS_Movement Files:**  
  Provided scenario files simulate realistic node mobility.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Dimitrios-Kafetzis/SDN-MANET-reactive-firewall-simulators.git
   cd SDN-MANET-reactive-firewall-simulators
   ```
2. **Install Dependencies:**
   Install the required Python libraries via pip:
   ```bash
   pip install networkx numpy matplotlib
   ```

## Usage
The repository includes two main simulation scripts:

### TCP/UDP Flood Attack Simulation:
Run the simulation that emulates TCP/UDP flood attacks and evaluates reactive mitigation:
```bash
python JCPIE_centralized_optimizer_simulation_-_TCP_UDP_flood_attacks.py
```
### Simplistic Multiple Runs Simulation:
Run multiple simulation iterations under varying conditions to evaluate optimization performance and overall mitigation delay:
```bash
python JCPIE_centralized_optimizer_simulation_simplistic_multiple_runs.py
```

## Configuration & Simulation Parameters
- **Mobility Models**:
Select the initial movement type (e.g., `random`, `convoy`, `small_teams`) to customize node mobility.
- **Attack Settings**:
Configure the number of malicious-victim pairs and packet counts to simulate different attack intensities.
- **Optimization Settings**:
Adjust parameters such as the maximum number of IDS nodes and the minimum coverage percentage to fine-tune the JCPIE optimization algorithms.

## Author
Dimitrios Kafetzis (kafetzis@aueb.gr and dimitrioskafetzis@gmail.com)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
