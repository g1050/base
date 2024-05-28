
def test_with():
    class MyContextManager:
        def __enter__(self):
            print("Entering the context")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print("Exiting the context")
            if exc_type:
                print(f"An exception occurred: {exc_val}")
            return True  # 处理异常，不再向外传播

    # with用于上下文管理
    with MyContextManager() as manager:
        print("Inside the context")
        raise ValueError("Something went wrong")

if __name__ == "__main__":
    test_with()