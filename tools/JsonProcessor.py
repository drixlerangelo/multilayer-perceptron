import json


class JsonProcessor:
    """Handles JSON-related tasks
    """

    @staticmethod
    def load(file):
        """Loads the json file provided the file name

        :param file: File name of the json file
        :type file: str

        :rtype: Any
        """

        with open(file, 'r') as json_file:
            return json.load(json_file)

    @staticmethod
    def save(file, data):
        """Save the json file provided the file name and data

        :param file: File name of the json file
        :type file: str

        :param data: The object that's going to be saved
        :type data: Any
        """

        with open(file, 'w') as json_file:
            json.dump(data, json_file)

    @staticmethod
    def beautify(file, data):
        """A beautified version in saving a JSON file

        :param file: File name of the json file
        :type file: str

        :param data: The object that's going to be saved
        :type data: Any
        """

        with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=4, sort_keys=True)
