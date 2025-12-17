from pydantic import BaseModel,EmailStr, Field
from typing import Optional

class Student(BaseModel):
    
    name:str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10, description='A decimal value represnting the cgpa of the student') # added contraint that cgpa will be greatter than 0 and less than 10
    
new_student = {'name':'aditya','email':'adityamohiteakm'} # it will throw validation error

student = Student(**new_student)

# also we can convert it to dictinory, json
student_dict = dict(student)

print(student_dict['name'])