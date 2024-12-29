def s_curve(
    height: float,
    mp: float,
    x: float,
    direction: str = 'down'
) -> float:
    '''
    Calculate an s-curve for discounting or ramping values

    Parameters:
    * height: The maximum value of the curve
    * mp: The midpoint of the curve
    * x: The x-value to calculate the curve for
    * direction: The direction of the curve, either 'down' or 'up'
    '''
    if direction == 'down':
        return (
            1 - (1 / (1 + 1.5 ** (
                (-1 * (x - mp)) *
                (10 / mp)
            )))
        ) * height
    else:
        return (1-(
            1 - (1 / (1 + 1.5 ** (
                (-1 * (x - mp)) *
                (10 / mp)
            )))
        )) * height