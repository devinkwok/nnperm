
class ScaleSpec:
    """
        axes_to_scale: str (name of layer): Tuple[ (for each dim in layer shape) Union[None (dim not permuted), str (name of scale assigned to dim)]]
        scale_to_axes: str (names of distinct scales): List[Tuple[str (name of layer with perm), int (dim with this perm)]]
    """