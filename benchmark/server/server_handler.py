from abc import ABC, abstractmethod


class ServerHandler(ABC):
    @abstractmethod
    def get_server_name():
        pass

    @abstractmethod
    def start_server():
        pass

    @abstractmethod
    def stop_server():
        pass

    @abstractmethod
    def handle_result(data, duration=None):
        pass

    @abstractmethod
    def get_result():
        pass
