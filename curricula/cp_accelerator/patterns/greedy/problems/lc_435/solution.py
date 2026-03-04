"""
Non-overlapping Intervals - Reference Solution
Time: O(n log n), Space: O(1)
"""


def eraseOverlapIntervals(intervals):
    """
    Find the minimum number of intervals to remove to make the rest non-overlapping.

    Args:
        intervals: List of [start, end] intervals

    Returns:
        Minimum number of intervals to remove

    Example:
        >>> eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]])
        1
    """
    if not intervals:
        return 0

    # Sort by end time (greedy: keep intervals that end earliest)
    intervals.sort(key=lambda x: x[1])

    count = 0
    prev_end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] < prev_end:
            # Overlap — remove the current interval (it ends later)
            count += 1
        else:
            # No overlap — keep this interval
            prev_end = intervals[i][1]

    return count


# Alias for compatibility with test runner
solve = eraseOverlapIntervals
