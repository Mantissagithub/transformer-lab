from typing import Any, Callable, Dict


class Registry:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(
                f"{self._kind} '{name}' not registered. Known: {sorted(self._items)}"
            )
        return self._items[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)


ATTENTION = Registry("attention")
FFN = Registry("feedforward")
NORM = Registry("normalization")
POS = Registry("positional")
CONNECTION = Registry("connection")
OPTIMIZER = Registry("optimizer")
SCHEDULER = Registry("scheduler")
LOSS = Registry("loss")
DATASET = Registry("dataset")
EMBEDDING = Registry("embedding")
PROJECTION = Registry("projection")
