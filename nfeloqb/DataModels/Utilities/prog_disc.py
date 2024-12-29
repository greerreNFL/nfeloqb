def prog_disc(
    obs: float,
    proj: float,
    scale: float,
    alpha: float
) -> float:
    '''
    Progressively discount a value as it moves away from zero, with an additional cap
    in place that limits how high the value can be.

    This is used to signal process an error values, where the long tail of the
    distribution (ie extreme departures from expectation) may provide less signal
    while simultaneously making week to week adjustments more volatile.

    Pre-processing an error in this way makes the EMA behave more like an
    Elo model, or bayesean updating process.

    Parameters:
    * obs: The observed value
    * proj: The projected value
    * scale: The scale of the error. Scale * 15 should align with where the long tail begins
    * alpha: The aggressiveness of the discounting. Reasonable values are between 0 and 0.005

    Returns:
    * The processed obs
    '''
    ## calculate the error ##
    abs_error = abs(obs - proj)
    error_direction = 1 if obs >= proj else -1
    ## control for instances with no error or discounting
    if abs_error == 0 or alpha == 0:
        return obs
    ## attempt to calc processed value while controlling for overflow errors ##
    try:
        return (
            proj +
            (
                error_direction *
                ## process error ##
                min(
                    abs_error, 0.309 * (alpha ** -0.864) * scale
                ) ** (
                    1 -
                    min(
                        (
                            min(abs_error, 0.309 * (alpha ** -0.864) * scale) /
                            scale
                        ) * alpha,
                        1
                    )
                )
            )
        )
    except OverflowError:
        return obs
