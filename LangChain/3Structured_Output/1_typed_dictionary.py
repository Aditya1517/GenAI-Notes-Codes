from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    
newPerson: Person = {"name": "Alice", "age": 30}

print(newPerson)