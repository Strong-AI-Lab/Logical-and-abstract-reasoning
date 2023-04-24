

class Model():

    def __init__(self, **kwargs):
        pass

    def load(self) -> None:
        raise NotImplementedError("Abstract model class does not implement load.")

    def answer_query(self, prompt : list) -> str:
        raise NotImplementedError("Abstract model class does not implement answer_query.")
    
    def format_data(self, data : dict) -> tuple:
        raise NotImplementedError("Abstract model class does not implement format_data.")
    
    def convert_input_list_to_text(self, input_list : list, separator = "\n", skip_instructions : bool = False) -> str:
        if skip_instructions:
            input_list = input_list[1:]
        
        return [separator.join([inp["content"][i] for inp in input_list]) for i in range(len(input_list[0]["content"]))]
        
