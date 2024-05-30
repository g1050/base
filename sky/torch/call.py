from typing import Any


class Person:
    def __init__(self) -> None:
        pass
    def __call__(self,name) -> Any:
        print(f"call {name}")
        
    def hello(self,name):
        print(f"Hello {name}")

person = Person()
person("zs") # 调用call方法
person.hello("ls")