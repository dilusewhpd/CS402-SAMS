"""Loads the student roster from the info.xml file."""

import xml.etree.ElementTree as ET
from models import Student


class StudentRepository:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path

    def load_students(self) -> list[Student]:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        students = [
            Student(
                no=s.find("no").text,
                student_id=s.find("id").text,
                name=s.find("name").text,
            )
            for s in root.findall("student")
        ]
        print(f"Total students loaded from XML: {len(students)}")
        return students