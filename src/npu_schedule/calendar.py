"""Earliest-fit Task calendar using exact clock ticks and indexed free intervals.

Finite endpoints and durations are exact integers (or Fractions in oracle tests).
The last free interval is unbounded. No tolerance is used for time comparisons.
"""

from dataclasses import dataclass
from math import inf


@dataclass
class Gap:
    low: float
    high: float
    priority: int
    left: "Gap | None" = None
    right: "Gap | None" = None
    maximum: float = 0

    def refresh(self):
        self.maximum = max(
            self.high - self.low,
            self.left.maximum if self.left else 0,
            self.right.maximum if self.right else 0,
        )
        return self


def split(root, key):
    if root is None:
        return None, None
    if root.low < key:
        root.right, right = split(root.right, key)
        return root.refresh(), right
    left, root.left = split(root.left, key)
    return left, root.refresh()


def merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if left.priority < right.priority:
        left.right = merge(left.right, right)
        return left.refresh()
    right.left = merge(left, right.left)
    return right.refresh()


def insert(root, node):
    if root is None:
        return node
    if node.priority < root.priority:
        node.left, node.right = split(root, node.low)
        return node.refresh()
    if node.low < root.low:
        root.left = insert(root.left, node)
    else:
        root.right = insert(root.right, node)
    return root.refresh()


def erase(root, key):
    if root.low == key:
        return merge(root.left, root.right)
    if key < root.low:
        root.left = erase(root.left, key)
    else:
        root.right = erase(root.right, key)
    return root.refresh()


def first_fit(root, release, duration):
    if root is None or root.maximum < duration:
        return None
    if root.low < release:
        if release + duration <= root.high:
            return root
        return first_fit(root.right, release, duration)
    earlier = first_fit(root.left, release, duration)
    if earlier is not None:
        return earlier
    if root.low + duration <= root.high:
        return root
    return first_fit(root.right, release, duration)


class TaskCalendar:
    def __init__(self, gap):
        self.gap = gap
        self.counter = 0
        self.root = self.node(0, inf)
        self.entries = []

    def node(self, low, high):
        # SplitMix64 priorities give deterministic shape without changing scheduling ties.
        self.counter += 1
        mask = (1 << 64) - 1
        value = (self.counter + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return Gap(low, high, value).refresh()

    def earliest(self, release, duration):
        interval = first_fit(self.root, release, duration)
        return max(release, interval.low), (interval.low, interval.high)

    def commit(self, start, end, unit, interval):
        low, high = interval
        self.root = erase(self.root, low)
        if low < start - self.gap:
            self.root = insert(self.root, self.node(low, start - self.gap))
        if end + self.gap < high:
            self.root = insert(self.root, self.node(end + self.gap, high))
        self.entries.append((start, end, unit))

    def order(self):
        return tuple(unit for _, _, unit in sorted(self.entries))
