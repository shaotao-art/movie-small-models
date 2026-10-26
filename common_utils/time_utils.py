def timecode_to_second(timecode: str) -> float:
    """Convert a timecode string to seconds.
    Args:
      timecode (str): Timecode string in the format "HH:MM:SS.mmm/HH:MM:SS,mmm".
    Returns:
      float: Time in seconds, or None if the input is invalid.
    """
    timecode = timecode.replace(",", ".")
    parts = timecode.split(":")
    if len(parts) != 3:
        return None
    hh = float(parts[0])
    mm = float(parts[1])
    ss = float(parts[2])
    return hh * 3600 + mm * 60 + ss


def second_to_timecode(seconds: float) -> str:
    """Convert seconds to a timecode string.
    Args:
      seconds (float): Time in seconds.
    Returns:
      str: Timecode string in the format "HH:MM:SS,mmm".
    """
    if seconds is None:
        return "00:00:00,000"
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02}:{mm:02}:{ss:06.3f}".replace(".", ",")


def time_iof(a: tuple[float, float], b: tuple[float, float]) -> float:
    """intersection over foreground of two time intervals
    a is the foreground"""
    start_a, end_a = a
    start_b, end_b = b
    if end_a <= start_b or end_b <= start_a:
        return 0.0
    intersection = min(end_a, end_b) - max(start_a, start_b)
    len_a = end_a - start_a
    return intersection / len_a


def time_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """intersection over union of two time intervals"""
    start_a, end_a = a
    start_b, end_b = b
    if end_a <= start_b or end_b <= start_a:
        return 0.0
    intersection = min(end_a, end_b) - max(start_a, start_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union
