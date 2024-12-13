#!/usr/bin/env python3
'''This calculates the derivative of a polynomial'''


def poly_derivative(poly):
    ''''''
    if not isinstance(poly, list) or len(
            poly) == 0 or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]
    derivative = [i * poly[i] for i in range(1, len(poly))]

    return derivative if derivative else [0]
