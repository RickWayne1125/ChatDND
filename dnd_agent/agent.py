from abc import abstractmethod

from openai import OpenAI
from config import OPENAI_API_KEY


class Agent:
    """
    Agent class
    This class is an extension from the OpenAI ChatGPT Assistants.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        # check if the assistants are already created
        self.assistant_id = self.check_assistant() or self.create_assistant()
        self.thread = self.client.beta.threads.create()  # this contains the session

    @abstractmethod
    def check_assistant(self) -> str:
        """
        Check if the assistants are already created
        :return: assistant id or None
        """
        pass

    @abstractmethod
    def create_assistant(self) -> str:
        """
        Create the assistants
        :return: assistant id
        """
        pass

    @abstractmethod
    def send_message(self, message: str) -> str:
        """
        Send a message to the assistants
        :param message: message to send
        :return: response from the assistants
        """
        pass
