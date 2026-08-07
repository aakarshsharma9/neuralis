import math
from abc import ABC, abstractmethod


# ==========================================
# 1. Class Creation & 3. Encapsulation
# ==========================================
class Student:
    """Base class representing a Student with encapsulated age attribute."""

    def __init__(self, name: str, age: int, grade: str):
        self.name = name
        self.__age = None  # Private attribute (Encapsulation)
        self.set_age(age)  # Use setter for validation/assignment
        self.grade = grade

    # Getter for private attribute __age
    def get_age(self) -> int:
        return self.__age

    # Setter for private attribute __age
    def set_age(self, age: int) -> None:
        if isinstance(age, int) and age > 0:
            self.__age = age
        else:
            raise ValueError("Age must be a positive integer.")

    # Method to display student information
    def display_info(self) -> str:
        return (
            f"Name: {self.name} | Age: {self.get_age()} | Grade: {self.grade}"
        )


# ==========================================
# 2. Inheritance
# ==========================================
class HighSchoolStudent(Student):
    """Subclass representing a High School Student inheriting from Student."""

    def __init__(self, name: str, age: int, grade: str, grade_level: str):
        # Initialize attributes from parent class
        super().__init__(name, age, grade)
        self.grade_level = grade_level  # Additional attribute

    # Overriding the display_info method (Polymorphic behavior via overriding)
    def display_info(self) -> str:
        base_info = super().display_info()
        return f"{base_info} | Grade Level: {self.grade_level}"


# ==========================================
# 4. Polymorphism
# ==========================================
def print_student_info(student: Student) -> None:
    """Function demonstrating polymorphism by executing display_info()

    on both Student and HighSchoolStudent instances uniformly.
    """
    print(student.display_info())


# ==========================================
# 5. Abstraction
# ==========================================
class Shape(ABC):
    """Abstract class defining a blueprint for geometric shapes."""

    @abstractmethod
    def calculate_area(self) -> float:
        """Abstract method to calculate area, implemented by subclasses."""
        pass


class Circle(Shape):
    """Concrete subclass of Shape representing a circle."""

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius

    def calculate_area(self) -> float:
        return math.pi * (self.radius**2)


class Rectangle(Shape):
    """Concrete subclass of Shape representing a rectangle."""

    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive.")
        self.width = width
        self.height = height

    def calculate_area(self) -> float:
        return self.width * self.height


# ==========================================
# Driver Code / Testing the Assignment Requirements
# ==========================================
if __name__ == "__main__":
    print("=== Testing 1, 2, 3 & 4: OOP Student Hierarchy & Polymorphism ===")

    # Creating base Student instance
    student1 = Student("Alice Smith", 15, "A")

    # Testing Encapsulation (Getter & Setter)
    print("Original Age:", student1.get_age())
    student1.set_age(16)
    print("Updated Age:", student1.get_age())

    # Creating HighSchoolStudent instance (Inheritance)
    hs_student = HighSchoolStudent(
        name="Bob Jones", age=17, grade="A+", grade_level="Senior"
    )

    print("\nDemonstrating Polymorphism with print_student_info():")
    print_student_info(student1)  # Invokes Student.display_info()
    print_student_info(hs_student)  # Invokes HighSchoolStudent.display_info()

    print("\n=== Testing 5: Abstraction & Subclasses ===")

    circle = Circle(radius=5.0)
    rectangle = Rectangle(width=4.0, height=6.0)

    print(f"Circle Area (r=5): {circle.calculate_area():.2f}")
    print(f"Rectangle Area (4x6): {rectangle.calculate_area():.2f}")