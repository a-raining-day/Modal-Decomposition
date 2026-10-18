"""2-D image extension of the proposed method (morphological envelope mean).

The 1-D method uses upper/lower cubic-spline envelopes.  A direct 2-D
spline/BEMD implementation is a separate research problem; this demo uses the
natural flat-structuring-element analogue:

    upper = grey_dilation(x, s)
    lower = grey_erosion(x, s)
    E(x)  = (upper + lower) / 2        # midrange / local morphological mean

and then applies the same add-noise + M-average + K-iteration loop.
"""
