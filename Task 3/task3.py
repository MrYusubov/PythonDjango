import pandas as pd
from dataclasses import dataclass


@dataclass
class InventoryItem:
    name: str
    qty: int
    price: float

    def __post_init__(self):
        if self.qty < 0:
            raise ValueError(f"Quantity can't be negative: {self.qty}")
        if self.price < 0:
            raise ValueError(f"Price can't be negative: {self.price}")


class Warehouse:
    __slots__ = ("_items",)

    def __init__(self):
        self._items: dict[str, InventoryItem] = {}

    @property
    def items(self):
        return self._items

    def add_item(self, item: InventoryItem):
        if item.name in self._items:
            existing = self._items[item.name]
            new_qty = existing.qty + item.qty
            self._items[item.name] = InventoryItem(
                name=existing.name,
                qty=new_qty,
                price=existing.price
            )
        else:
            self._items[item.name] = item

    def remove_item(self, name: str, qty: int = None):
        if name not in self._items:
            raise KeyError(f"{name} not found")

        if qty is None or qty >= self._items[name].qty:
            del self._items[name]
        else:
            existing = self._items[name]
            self._items[name] = InventoryItem(
                name=existing.name,
                qty=existing.qty - qty,
                price=existing.price
            )

    def total_value(self) -> float:
        return sum(item.qty * item.price for item in self._items.values())

    @classmethod
    def from_excel(cls, path: str, sheet_name: str = "InventoryItems"):
        warehouse = cls()
        df = pd.read_excel(path, sheet_name=sheet_name)

        for _, row in df.iterrows():
            item = InventoryItem(
                name=row["name"],
                qty=int(row["qty"]),
                price=float(row["price"])
            )
            warehouse.add_item(item)

        return warehouse

w = Warehouse.from_excel("oopData.xlsx", sheet_name="InventoryItems")
print("Items:", w.items)
print("Total value:", w.total_value())
