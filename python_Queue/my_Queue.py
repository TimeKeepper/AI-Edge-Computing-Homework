class ring_Queue(object):
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size
        self.head = 0
        self.tail = 0
        self.count = 0

    def lenth(self):
        return self.count

    def is_empty(self):
        return self.lenth() == 0

    def is_full(self):
        return self.lenth() == self.size

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full")
            return
        self.queue[self.tail] = item
        self.tail = (self.tail + 1) % self.size
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return
        item = self.queue[self.head]
        self.head = (self.head + 1) % self.size
        self.count -= 1
        return item

