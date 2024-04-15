import colorama

RED = colorama.Fore.RED
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Style.RESET_ALL


# def tqdm_params(desc, prog_bar, leave=False, position=1, silent=False):
#     return {
#         "desc": f"{desc:<25}",
#         "disable": ((not prog_bar) or silent),
#         "position": position,
#         "leave": leave,
#         # "ascii": True,
#         # "ncols": 120
#     }
