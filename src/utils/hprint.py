# src/utils/hprint.py

class TextStyle:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def hprint(text: str, color: str = TextStyle.RED, bold: bool = True):
    """
    강조된 텍스트를 출력하는 함수.

    Args:
        text (str): 출력할 텍스트.
        color (str): 텍스트 색상 (기본값: RESET).
        bold (bool): 텍스트를 굵게 표시할지 여부 (기본값: True).
    """
    style = f"{color}"
    if bold:
        style += TextStyle.BOLD
    print(f"\n{style}{text}{TextStyle.RESET}\n")
    
