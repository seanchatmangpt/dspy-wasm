"""Pure-Python rpds projection for WASI.

``rpds-py`` is a Rust extension with no WASI build. DSPy reaches it eagerly via
``jsonschema -> referencing``. This projection implements the persistent
(immutable, structurally-updated) collection surface those libraries use with
copy-on-write semantics. Every mutator returns a new collection; the receiver
is never modified.
"""

from __future__ import annotations

from collections.abc import Hashable, ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from collections.abc import Set as AbstractSet
from typing import Any

__all__ = ["HashTrieMap", "HashTrieSet", "List"]


class HashTrieMap(Mapping):
    __slots__ = ("_data", "_hash")

    def __init__(self, value: Mapping | Iterable[tuple[Any, Any]] = (), **kwds: Any) -> None:
        data = dict(value)
        data.update(kwds)
        self._data = data
        self._hash: int | None = None

    @classmethod
    def convert(cls, value: Any) -> HashTrieMap:
        return value if isinstance(value, HashTrieMap) else cls(value)

    @classmethod
    def fromkeys(cls, keys: Iterable[Any], value: Any = None) -> HashTrieMap:
        return cls((key, value) for key in keys)

    def _derive(self, data: dict) -> HashTrieMap:
        new = HashTrieMap.__new__(HashTrieMap)
        new._data = data
        new._hash = None
        return new

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"HashTrieMap({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self._data == dict(other.items())
        return NotImplemented

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __reduce__(self):
        return (HashTrieMap, (self._data,))

    def keys(self) -> KeysView:
        return self._data.keys()

    def values(self) -> ValuesView:
        return self._data.values()

    def items(self) -> ItemsView:
        return self._data.items()

    def insert(self, key: Any, value: Any) -> HashTrieMap:
        data = dict(self._data)
        data[key] = value
        return self._derive(data)

    def remove(self, key: Any) -> HashTrieMap:
        if key not in self._data:
            raise KeyError(key)
        data = dict(self._data)
        del data[key]
        return self._derive(data)

    def discard(self, key: Any) -> HashTrieMap:
        return self.remove(key) if key in self._data else self

    def update(self, *maps: Mapping | Iterable[tuple[Any, Any]], **kwds: Any) -> HashTrieMap:
        data = dict(self._data)
        for each in maps:
            data.update(each)
        data.update(kwds)
        return self._derive(data)


class HashTrieSet(AbstractSet, Hashable):
    __slots__ = ("_data",)

    def __init__(self, value: Iterable[Any] = ()) -> None:
        self._data = frozenset(value)

    @classmethod
    def _from_iterable(cls, it: Iterable[Any]) -> HashTrieSet:
        return cls(it)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return hash(self._data)

    def __repr__(self) -> str:
        return f"HashTrieSet({set(self._data)!r})"

    def __reduce__(self):
        return (HashTrieSet, (tuple(self._data),))

    def insert(self, value: Any) -> HashTrieSet:
        return HashTrieSet(self._data | {value})

    def remove(self, value: Any) -> HashTrieSet:
        if value not in self._data:
            raise KeyError(value)
        return HashTrieSet(self._data - {value})

    def discard(self, value: Any) -> HashTrieSet:
        return HashTrieSet(self._data - {value}) if value in self._data else self

    def update(self, *iterables: Iterable[Any]) -> HashTrieSet:
        data = set(self._data)
        for each in iterables:
            data.update(each)
        return HashTrieSet(data)

    def union(self, *iterables: Iterable[Any]) -> HashTrieSet:
        return self.update(*iterables)

    def intersection(self, *iterables: Iterable[Any]) -> HashTrieSet:
        return HashTrieSet(self._data.intersection(*iterables))

    def difference(self, *iterables: Iterable[Any]) -> HashTrieSet:
        return HashTrieSet(self._data.difference(*iterables))

    def symmetric_difference(self, other: Iterable[Any]) -> HashTrieSet:
        return HashTrieSet(self._data.symmetric_difference(other))


class List(Hashable):
    """Persistent singly-linked list; ``push_front`` is the cheap operation."""

    __slots__ = ("_items",)

    def __init__(self, *values: Any) -> None:
        if (
            len(values) == 1
            and not isinstance(values[0], (str, bytes))
            and hasattr(values[0], "__iter__")
        ):
            self._items = tuple(values[0])
        else:
            self._items = tuple(values)

    @classmethod
    def _of(cls, items: tuple) -> List:
        new = cls.__new__(cls)
        new._items = items
        return new

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, List):
            return self._items == other._items
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("rpds.List", self._items))

    def __repr__(self) -> str:
        return f"List({list(self._items)!r})"

    def __reversed__(self) -> Iterator[Any]:
        return reversed(self._items)

    def __reduce__(self):
        return (List, (list(self._items),))

    @property
    def first(self) -> Any:
        if not self._items:
            raise IndexError("empty list has no first element")
        return self._items[0]

    @property
    def rest(self) -> List:
        return List._of(self._items[1:])

    def push_front(self, value: Any) -> List:
        return List._of((value, *self._items))

    def drop_first(self) -> List:
        if not self._items:
            raise IndexError("empty list has no first element")
        return List._of(self._items[1:])
