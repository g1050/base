
def test():   
    def simple_generator():
        yield 1
        yield 2
        yield 3

    gen = simple_generator()
    print(next(gen))  
    print(next(gen))  
    print(next(gen))  

    for num in simple_generator():
        print(num)

    def generator_with_send():
        value = yield "start"
        while True:
            value = yield value

    gen = generator_with_send()
    print(next(gen))       
    print(gen.send(10))    
    print(gen.send(20))    
def main():
    class Counter:
        def __init__(self,low,high) -> None:
            self.current = low
            self.high = high
        def __iter__(sefl):
            return self
        def __next__(self):
            if self.current > self.high:
                raise StopIteration
            else:
                self.current += 1
                return self.current -1
    
    counter = Counter(low=1,high=10)
    print(next(counter))
    print(next(counter))
    print(next(counter))
    print(next(counter))

if __name__ == "__main__":
    main()