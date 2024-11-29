# ML-Learn Examples

## 요구사항
vscode 환경에서 아래의 확장 프로그램을 설치해주세요.
- Python
- Jupyter

## 설치

```bash
# poetry 설치
pip install poetry

# (선택) poetry 가상환경 경로 설정 (현재 디렉토리에 가상환경을 생성)
poetry config virtualenvs.in-project true

# 종속성 잠금 파일 생성
poetry lock

# 종속성 설치
poetry install

# 가상 환경 활성화
poetry shell
```

## 노트북 실행

- ML 예제와 관련된 노트북은 `notebooks` 디렉토리에 있습니다.
- .ipynb 파일을 VSCode에서 실행하거나, Jupyter Notebook에서 실행할 수 있습니다.
- 코드 실행전 poetry에서 설치한 가상환경 인터프리터를 사용하도록 설정해주세요.(.venv/Scripts/python)
