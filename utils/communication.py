"""
Client-server communication utilities using multiprocessing queues.
"""

from multiprocessing import Queue

class CommunicationChannel:
    """
    Holds queues for communication between clients and server.
    One queue per client for sending messages.
    One queue for server to send aggregated responses.
    """

    def __init__(self, num_clients):
        self.client_queues = [Queue() for _ in range(num_clients)]
        self.server_queue = Queue()

    def send_to_server(self, client_id, message):
        self.client_queues[client_id].put(message)

    def receive_from_client(self, client_id):
        return self.client_queues[client_id].get()

    def send_to_client(self, message):
        self.server_queue.put(message)

    def receive_from_server(self):
        return self.server_queue.get()
