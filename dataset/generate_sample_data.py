"""
STEP 0: Sample Dataset Generator
---------------------------------
Ye sirf TESTING ke liye hai - taaki code turant chal ke dikhaye.
REAL project ke liye Kaggle se asli dataset download karo:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Download karke True.csv aur Fake.csv isi 'dataset/' folder mein daal do,
phir ye sample generator ki zaroorat nahi padegi.
"""

import pandas as pd
import random

real_templates = [
    "Government announces new {topic} policy for {year}",
    "Scientists at {org} publish study on {topic}",
    "{country} reports economic growth of {num}% in Q{q}",
    "Local officials confirm infrastructure project in {city}",
    "Central bank keeps interest rates unchanged this quarter",
    "Ministry releases annual report on {topic} development",
    "Researchers find link between {topic} and public health",
    "Election commission announces schedule for upcoming polls",
    "Company reports quarterly earnings in line with expectations",
    "University study shows steady progress in {topic} sector",
]

fake_templates = [
    "SHOCKING: {topic} secretly controlled by {org}, insiders reveal!!!",
    "You won't BELIEVE what {country} is hiding about {topic}",
    "Doctors HATE this one weird trick to cure {topic} overnight",
    "BREAKING: {city} government caught faking {topic} data",
    "Aliens linked to {topic} conspiracy, experts SILENCED",
    "Miracle cure for {topic} banned by big pharma, leaked docs show",
    "Celebrity secretly reveals truth about {topic}, media covers up",
    "URGENT: {country} on verge of collapse, mainstream media SILENT",
    "Scientists FIRED for exposing real truth about {topic}",
    "This ONE weird fact about {topic} they don't want you to know",
]

topics = ["economy", "healthcare", "education", "climate", "technology", "agriculture", "energy", "transport"]
orgs = ["WHO", "United Nations", "World Bank", "IMF", "local university", "research institute"]
countries = ["India", "USA", "UK", "Germany", "Japan", "Brazil"]
cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Pune"]

def fill(template):
    return template.format(
        topic=random.choice(topics), org=random.choice(orgs),
        country=random.choice(countries), city=random.choice(cities),
        year=random.choice([2024, 2025, 2026]), num=random.randint(2, 9),
        q=random.randint(1, 4)
    )

random.seed(42)
real_rows = [{"title": fill(random.choice(real_templates)),
              "text": fill(random.choice(real_templates)) + ". " + fill(random.choice(real_templates)),
              "subject": "news", "date": "2025-01-01"} for _ in range(150)]

fake_rows = [{"title": fill(random.choice(fake_templates)),
              "text": fill(random.choice(fake_templates)) + ". " + fill(random.choice(fake_templates)),
              "subject": "news", "date": "2025-01-01"} for _ in range(150)]

pd.DataFrame(real_rows).to_csv("dataset/True.csv", index=False)
pd.DataFrame(fake_rows).to_csv("dataset/Fake.csv", index=False)
print("Sample True.csv and Fake.csv created (150 rows each) for testing.")
