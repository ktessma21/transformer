import math

def lr_cosine_schedule(
        t: float,
        amax: float,
        amin: float,
        tW: float,
        tC: float
    ) -> float:

    if t < tW:
        return amax * t / tW

    if t <= tC:
        return amin + 0.5 * (amax - amin) * (
            1 + math.cos(math.pi * (t - tW) / (tC - tW))
        )

    return amin