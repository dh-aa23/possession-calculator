import numpy as np
def get_centre(bbox):
    """
    Calculate the center of a bounding box.

    Args:
        bbox (list): A list containing the coordinates of the bounding box in the format [x1, y1, x2, y2].

    Returns:
        tuple: A tuple containing the x and y coordinates of the center of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_width(bbox):
    """
    Calculate the width of a bounding box.

    Args:
        bbox (list): A list containing the coordinates of the bounding box in the format [x1, y1, x2, y2].

    Returns:
        int: The width of the bounding box.
    """
    x1, _, x2, _ = bbox
    return x2 - x1

def measure_distace(p1,p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)