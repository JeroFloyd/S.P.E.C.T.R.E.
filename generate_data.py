from __future__ import annotations
import os
import random
from datetime import datetime, timedelta
import pandas as pd

PRODUCTS = [
    "Laptop Pro 15", "Wireless Mouse", "USB-C Hub", "Mechanical Keyboard",
    "4K Monitor", "Webcam HD", "Standing Desk", "Ergonomic Chair",
    "Noise Cancelling Headphones", "External SSD 1TB", "Smart Watch",
    "Tablet 10in", "Portable Charger", "LED Desk Lamp", "Graphics Card RTX",
]
STATUSES = ["completed", "pending", "cancelled", "refunded", "processing"]
STATUS_WEIGHTS = [0.55, 0.20, 0.10, 0.08, 0.07]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d"]
REGIONS = ["North", "South", "East", "West", "Central"]
BAD_STATUSES = ["COMPLETED", "unknown", "N/A"]

def make_batch(batch_num: int, n_rows: int = 80) -> pd.DataFrame:
    rng = random.Random(42 + batch_num * 7)
    rows = []
    for i in range(n_rows):
        order_id = f"ORD-{batch_num * 1000 + i:05d}"
        customer_id = f"CUST-{rng.randint(1000, 9999)}"
        product = rng.choice(PRODUCTS)
        quantity = rng.randint(1, 20)
        unit_price = round(rng.uniform(9.99, 999.99), 2)
        status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]
        region = rng.choice(REGIONS)
        sales_rep = f"REP-{rng.randint(100, 199)}"
        base_date = datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 730))
        date_str = base_date.strftime(rng.choice(DATE_FORMATS))
        
        if i > 0 and rng.random() < 0.03:
            order_id = f"ORD-{batch_num * 1000 + i - 1:05d}"
        if rng.random() < 0.04:
            quantity = ""
        if rng.random() < 0.02:
            unit_price = -abs(unit_price)
        if rng.random() < 0.02:
            status = rng.choice(BAD_STATUSES)
        if rng.random() < 0.02:
            date_str = "not-a-date"
        
        rows.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "status": status,
            "order_date": date_str,
            "region": region,
            "sales_rep": sales_rep,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    for i in range(1, 7):
        df = make_batch(batch_num=i)
        path = f"data/raw/orders_{i}.csv"
        df.to_csv(path, index=False)
        print(f"  orders_{i}.csv  ->  {len(df)} rows")
    print("\nDone — 6 files written to data/raw/")
