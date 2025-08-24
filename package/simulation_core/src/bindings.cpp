#include <nanobind/nanobind.h>

// A simple C++ function to be exposed to Python.
int add(int a, int b) { return a + b; }

// The NB_MODULE macro defines the entry point for the Python module.
NB_MODULE(simulation_core, m) {
  m.doc() = "A high-performance C++ simulation core.";
  // Expose the 'add' function to Python.
  m.def("add", &add, "A function that adds two numbers.");
}