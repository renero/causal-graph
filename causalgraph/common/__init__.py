import colorama

RED = colorama.Fore.RED
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Style.RESET_ALL

def tqdm_params(desc, progbar):
   return {
    "desc": f"{desc:<25}",
    "disable": not progbar, 
    "position": 1, 
    "leave": True
}