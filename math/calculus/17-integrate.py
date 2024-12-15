#!/usr/bin/env python3
"""This function calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    '''calculates the integral of a polynomial

    Args:
        poly (list): coefficients representing a polynomial
        C (int, optional): represents the integration constant. Defaults to 0.

    Returns:
        list: coefficients representing the integral of the polynomial
    '''
    # Validate poly
    if (not isinstance(poly, list) or len(poly) == 0 or
            not all(isinstance(c, (int, float)) for c in poly)):
        return None

    # Validate C
    if not isinstance(C, int):
        return None

    # Perform integration
    # The resulting polynomial will have one more term
    # The first term is C (the integration constant)
    result = [C]
    for i, coeff in enumerate(poly):
        # Avoid division by zero; this never happens since i+1 >= 1
        new_coeff = coeff / (i+1)
        # If the new_coeff is effectively an integer, convert it to int
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)
        result.append(new_coeff)

    # Remove trailing zeros from the result if there are any
    # But ensure at least one coefficient remains
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
