# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import unittest
import numpy

# Package import
import pysap

class TestAdjointOperatorWaveletTransform(unittest.TestCase):
    """ Test the adjoint operator of the NFFT both for 2D and 3D.
    """
    def setUp(self):
        """ Set the number of iterations.
        """
        self.transforms = pysap.wavelist()
        self.N = 64
        self.max_iter = 20
        self.nb_scale = 4

    def test_Wavelet2D_ISAP(self):
        """Test the adjoint operator for the 2D Wavelet transform
        """
        for transform in self.transforms['isap-2d']:
            print("Process Wavelet_ISAP_2D" + transform)
            wavelet_op_adj = pysap.load_transform(transform)
            transform = wavelet_op_adj(
                nb_scale=self.nb_scale, verbose=0)
            Img = (numpy.random.randn(self.N, self.N) +
                   1j * numpy.random.randn(self.N, self.N))
            transform.data = Img
            transform.analysis()
            f_p = numpy.asarray(transform.analysis_data)
            f = (numpy.random.randn(*f_p.shape) +
                 1j * numpy.random.randn(*f_p.shape))
            transform.analysis_data = f
            I_p = transform.synthesis()
            x_d = numpy.dot(Img.flatten(), numpy.conj(I_p).flatten())
            x_ad = numpy.dot(f_p.flatten(), numpy.conj(f).flatten())
            mismatch = (1. - numpy.mean(
                numpy.isclose(x_d, x_ad,
                              rtol=1e-6)))
            print("      mismatch = ", mismatch)
            self.assertTrue(numpy.isclose(x_d, x_ad, rtol=1e-6))
        print(" Wavelet2 adjoint test passes")


if __name__ == "__main__":
    unittest.main()
