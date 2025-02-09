#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for std::vector support
#include "fitps.h"

namespace py = pybind11;

PYBIND11_MODULE(fitps, m)
{
     m.doc() = "Python bindings for FITPS";

     py::class_<FITPS>(m, "FITPS")
         .def(py::init<int, int, int>(), py::arg("cycle_size"), py::arg("buffer_size"), py::arg("thresh"),
              "Constructor requiring cycle_size and buffer_size")
         .def("add_samples", &FITPS::add_samples, py::arg("volt_sample"), py::arg("amp_sample"),
              "Adds voltage and current samples, returns transformed cycle data")
         .def("clear", &FITPS::clear, "Clears internal buffers")
         .def("transform", &FITPS::transform,
              py::arg("volts"), py::arg("amps"), py::arg("locs") = std::vector<int>{},
              "Transforms the whole voltage and current vectors with an optional locs parameter");
}
