# Pill Project
🗓 프로젝트 소개 : Pill Project</br>
🗓 기간 : 2022.9.24 ~   </br>
🗓 팀원:  [준석](https://github.com/dotdotot)</br>
🗓 리뷰어: [준석](https://github.com/dotdotot)</br></br>

# Yolov5
yolo 커스텀 데이터셋 학습시키기(colab)</br>

1. 드라이브 마운트</br>
<code>
    #드라이브 마운트</br>

    from google.colab import drive

    drive.mount('/content/drive')
</code></br></br>

2. 환경 세팅</br>

내 구글 드라이브로 이동</br>
%cd "/content/drive/MyDrive"</br>
Yolov5 github 레포지토리 clone</br>
!git clone https://github.com/ultralytics/yolov5.git</br>
필요한 모듈 설치</br>
!pip install -U -r yolov5/requirements.txt</br>
</br>

import torch</br>
#파이토치 버전 확인, cuda device properties 확인</br>
print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))</br>
</br>
런타임 -> 런타임 유형 변경에 들어가서 하드웨어 가속기 GPU로 변경</br></br>
이후 custom 모델에 사용할 데이터 수집하기</br></br>

3. data.yaml 파일 생성</br>
data.yaml : 모델 학습을 위한 작은 튜토리얼 같은 것 (내 드라이브/dataset 아래에 만들어준다)</br>
<img src ="C:\\vsCode\PillProject\image\다운로드.png">

# Color

# Shape

# String

# 사용한 라이브러리
* yolov5

color에서 사용한 라이브러리
* numpy
* cv2
* KMeans
* matplotlib
* PIL
* os

colorCss에서 사용한 라이브러리
* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* os
* PIL
* shutil

shape에서 사용한 라이브러리
* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* os
* PIL
* shutil

string에서 사용한 라이브러리
* Image
* pytesseract

# Commit 규칙
> 커밋 제목은 최대 50자 입력 </br>
본문은 한 줄 최대 72자 입력 </br>
Commit 메세지 </br>

🪛[chore]: 코드 수정, 내부 파일 수정. </br>
✨[feat]: 새로운 기능 구현. </br>
🎨[style]: 스타일 관련 기능.(코드의 구조/형태 개선) </br>
➕[add]: Feat 이외의 부수적인 코드 추가, 라이브러리 추가 </br>
🔧[file]: 새로운 파일 생성, 삭제 시 </br>
🐛[fix]: 버그, 오류 해결. </br>
🔥[del]: 쓸모없는 코드/파일 삭제. </br>
📝[docs]: README나 WIKI 등의 문서 개정. </br>
💄[mod]: storyboard 파일,UI 수정한 경우. </br>
✏️[correct]: 주로 문법의 오류나 타입의 변경, 이름 변경 등에 사용합니다. </br>
🚚[move]: 프로젝트 내 파일이나 코드(리소스)의 이동. </br>
⏪️[rename]: 파일 이름 변경이 있을 때 사용합니다. </br>
⚡️[improve]: 향상이 있을 때 사용합니다. </br>
♻️[refactor]: 전면 수정이 있을 때 사용합니다. </br>
🔀[merge]: 다른브렌치를 merge 할 때 사용합니다. </br>
✅ [test]: 테스트 코드를 작성할 때 사용합니다. </br>