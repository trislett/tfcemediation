import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

#    Fast TFCE algorithm using prior adjacency sets
#    Copyright (C) 2016,2025  Lea Waller, Tristram Lett

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

cdef extern from "libs/fast_tfce.hpp":
    void tfce[T](float H, float E, float minT, float deltaT, vector[vector[int]]& adjacencyList, T* image, T* enhn)

cdef class CreateAdjSet:
    cdef vector[vector[int]] *Adjacency
    cdef float H
    cdef float E

    def __init__(self, float H, float E, pyAdjacency):
        self.H = H
        self.E = E

        # Allocate and populate the adjacency list
        self.Adjacency = new vector[vector[int]]()
        cdef vector[int] Adjacency__
        for i in range(len(pyAdjacency)):
            Adjacency__ = pyAdjacency[i]
            self.Adjacency.push_back(Adjacency__)

    def __dealloc__(self):
        # Free the allocated memory
        del self.Adjacency

    def run(self, np.ndarray[float, ndim=1, mode="c"] image not None, 
                  np.ndarray[float, ndim=1, mode="c"] enhn not None):
        # Ensure the input arrays have the same size
        if image.shape[0] != enhn.shape[0]:
            raise ValueError("image and enhn must have the same length")

        # Call the optimized C++ function
        tfce[float](self.H, self.E, 0, 0, self.Adjacency[0], &image[0], &enhn[0])
