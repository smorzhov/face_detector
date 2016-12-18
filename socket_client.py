import socket


class SocketClient:
    """Simple socket client class"""
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send(self, host, port, message):
        """It sends a message to a socket server"""
        self.client.connect((host, port))
        self.client.send(message)
