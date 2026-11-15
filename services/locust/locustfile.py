import random
from locust import HttpUser, task, between


class RealEstateLoadTester(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict_property(self):
        payload = {
            "brokered_by": str(random.choice([103378, 52707, 99999])),
            "status": random.choice(["for_sale", "ready_to_build"]),
            "bed": random.randint(1, 6),
            "bath": random.randint(1, 5),
            "acre_lot": round(random.uniform(0.05, 2.5), 2),
            "street": str(random.randint(100000, 999999)),
            "city": random.choice(["San Juan", "Ponce", "Mayaguez", "NewCityTest"]),
            "state": random.choice(["Puerto Rico", "Virgin Islands", "Massachusetts"]),
            "zip_code": random.choice(["00901", "00716", "00680"]),
            "house_size": round(random.uniform(650, 4500), 1),
            "prev_sold_date": "2020-08-15",
        }
        self.client.post("/predict", json=payload)

    @task(1)
    def check_health(self):
        self.client.get("/health")
