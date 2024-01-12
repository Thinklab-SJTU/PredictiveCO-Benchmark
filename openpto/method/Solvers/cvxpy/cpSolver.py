#!/usr/bin/env python
# coding: utf-8
"""
Abstract optimization model based on GurobiPy
"""


class optCPSolver:
    """ """

    def __init__(self, modelSense):
        self.modelSense = modelSense
