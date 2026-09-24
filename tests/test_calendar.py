import random
from fractions import Fraction

import pytest

from npu_schedule.calendar import TaskCalendar
from npu_schedule.construction import insertion


@pytest.mark.parametrize("gap", [0, 100])
@pytest.mark.parametrize("seed", [11, 47, 20260924])
def test_indexed_calendar_matches_exhaustive_first_fit(gap, seed):
    rng = random.Random(seed)
    indexed = TaskCalendar(gap)
    reference = []
    for task in range(1500):
        release = rng.randrange(0, 100000)
        duration = rng.randrange(1, 500)
        if gap == 0:
            release = Fraction(release, 60)
            duration = Fraction(duration, 60)
        start, position = insertion(reference, release, duration, gap)
        chosen, interval = indexed.earliest(release, duration)
        assert chosen == start
        reference.insert(position, (start, start + duration, task))
        indexed.commit(chosen, chosen + duration, task, interval)
    assert indexed.order() == tuple(u for _, _, u in reference)
